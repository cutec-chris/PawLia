"""
Picoclaw custom Radicale storage backend.

URL structure:
  /<user>/calendar/           → session/<user>/workspace/calendar/*.md (YAML-frontmatter Markdown)
  /<user>/calendar/<href>.ics → single event (converted on-the-fly)
  /<user>/contacts/           → session/<user>/workspace/contacts/*.md (YAML-frontmatter Markdown)
  /<user>/contacts/<href>.vcf → single contact (converted on-the-fly)

Calendar markdown format:
  ---
  title: Meeting
  date: 2026-05-05
  allDay: false
  startTime: "10:00"
  endTime: "11:00"
  location: Office
  uid: <uuid>
  ---
  Description text

Contact markdown format:
  ---
  uid: <uuid>
  name: Max Mustermann
  firstname: Max
  lastname: Mustermann
  email: max@example.com          # or list for multiple
  phone: "+49 123 456"            # or list for multiple
  organization: TCSAG
  job_title: Developer
  birthday: "1990-01-01"
  address: "Musterstr. 1, Berlin"
  url: https://example.com
  ---
  Notes / Freitext

Filename convention: contacts/<name>.md, calendar/<YYYY-MM-DD title>.md
"""

import hashlib
import os
import re
import threading
import unicodedata
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Iterator, Mapping, Optional, Tuple

import uuid

import vobject
import yaml
from icalendar import Calendar, Event, vDate, vDatetime

import radicale.item as radicale_item
from radicale.storage import BaseCollection, BaseStorage

_CONFIG_PATH = os.environ.get("PICOCLAW_CONFIG", "/config.yaml")
_config_cache: dict = {}
_config_mtime: float = 0.0


def _load_config() -> dict:
    global _config_cache, _config_mtime
    try:
        mtime = os.path.getmtime(_CONFIG_PATH)
        if mtime != _config_mtime:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                _config_cache = yaml.safe_load(f) or {}
            _config_mtime = mtime
    except Exception:
        pass
    return _config_cache


def _resolve_user_id(login: str) -> str:
    """Map a Radicale login name to a picoclaw user_id via caldav.users in config.yaml.
    Falls back to the login name itself if no mapping is configured."""
    cfg = _load_config()
    mapping = (cfg.get("caldav") or {}).get("users") or {}
    return mapping.get(login, login)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_filename(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r'[^\w\s\-]', "", s)
    return re.sub(r'\s+', " ", s).strip()[:64]


def _etag(content: str) -> str:
    return f'"{hashlib.md5(content.encode()).hexdigest()}"'


def _mtime_rfc7231(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")


# ---------------------------------------------------------------------------
# Calendar: .md ↔ iCal conversion
# ---------------------------------------------------------------------------

def _md_to_ical(filepath: str, href: str) -> Optional[str]:
    """Read a picoclaw event .md file, return a full VCALENDAR iCal string."""
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None

    if not text.lstrip().startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return None

    body = parts[2].strip()

    uid = fm.get("uid") or href.removesuffix(".ics")
    title = fm.get("title", "Untitled")
    date_str = str(fm.get("date", ""))
    all_day = fm.get("allDay", False)
    start_time = str(fm.get("startTime", "")) if fm.get("startTime") else ""
    end_time = str(fm.get("endTime", "")) if fm.get("endTime") else ""
    end_date_str = str(fm.get("endDate", date_str)) if fm.get("endDate") else date_str
    location = fm.get("location", "")

    try:
        if all_day or not start_time:
            start_dt: date | datetime = date.fromisoformat(date_str)
            dtstart = vDate(start_dt)
        else:
            start_dt = datetime.fromisoformat(f"{date_str}T{start_time}:00")
            dtstart = vDatetime(start_dt)
    except Exception:
        return None

    dtend = None
    if end_time:
        try:
            dtend = vDatetime(datetime.fromisoformat(f"{end_date_str}T{end_time}:00"))
        except Exception:
            pass
    elif all_day:
        try:
            end_d = date.fromisoformat(end_date_str) + timedelta(days=1)
            dtend = vDate(end_d)
        except Exception:
            pass

    cal = Calendar()
    cal.add("prodid", "-//Picoclaw//Picoclaw CalDAV//EN")
    cal.add("version", "2.0")

    ev = Event()
    ev.add("uid", uid)
    ev.add("summary", title)
    ev.add("dtstart", dtstart)
    if dtend is not None:
        ev.add("dtend", dtend)
    if location:
        ev.add("location", location)
    if body:
        ev.add("description", body)
    ev.add("dtstamp", datetime.now(timezone.utc))
    cal.add_component(ev)

    return cal.to_ical().decode("utf-8")


def _ical_to_md(ical_text: str) -> Optional[Tuple[str, dict]]:
    """
    Parse a VCALENDAR iCal string → (md_content, meta_dict).
    meta_dict contains: title, date_str, uid, start_iso, end_iso.
    Returns None on parse failure.
    """
    try:
        cal = Calendar.from_ical(ical_text)
    except Exception:
        return None

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        uid = str(component.get("UID", ""))
        summary = str(component.get("SUMMARY", "Untitled"))
        dtstart_prop = component.get("DTSTART")
        dtend_prop = component.get("DTEND")
        description = str(component.get("DESCRIPTION", ""))
        location = str(component.get("LOCATION", ""))

        if dtstart_prop is None:
            return None

        dtstart = dtstart_prop.dt
        all_day = isinstance(dtstart, date) and not isinstance(dtstart, datetime)

        if all_day:
            date_str = dtstart.strftime("%Y-%m-%d")
            start_time = None
            end_time = None
            end_date = None
        else:
            if hasattr(dtstart, "tzinfo") and dtstart.tzinfo is not None:
                dtstart = dtstart.astimezone(tz=None).replace(tzinfo=None)
            date_str = dtstart.strftime("%Y-%m-%d")
            start_time = dtstart.strftime("%H:%M")
            end_time = None
            end_date = None
            if dtend_prop:
                dtend = dtend_prop.dt
                if isinstance(dtend, datetime):
                    if dtend.tzinfo is not None:
                        dtend = dtend.astimezone(tz=None).replace(tzinfo=None)
                    end_time = dtend.strftime("%H:%M")
                    end_date_s = dtend.strftime("%Y-%m-%d")
                    if end_date_s != date_str:
                        end_date = end_date_s

        fm: dict = {
            "title": summary,
            "date": date_str,
            "allDay": all_day,
            "type": "single",
            "uid": uid,
        }
        if not all_day and start_time:
            fm["startTime"] = start_time
        if not all_day and end_time:
            fm["endTime"] = end_time
        if end_date:
            fm["endDate"] = end_date
        if location:
            fm["location"] = location

        content = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).rstrip()}\n---\n"
        if description:
            content += f"\n{description}\n"

        meta = {
            "title": summary,
            "date_str": date_str,
            "uid": uid,
            "start_iso": f"{date_str}T{start_time}:00" if start_time else date_str,
            "end_iso": f"{(end_date or date_str)}T{end_time}:00" if end_time else "",
        }
        return content, meta

    return None


# ---------------------------------------------------------------------------
# Contacts: .md ↔ vCard conversion
# ---------------------------------------------------------------------------

def _md_to_vcf(filepath: str, href: str) -> Optional[str]:
    """Read a contact .md file, return a vCard string."""
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None

    if not text.lstrip().startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return None
    notes = parts[2].strip()

    uid = fm.get("uid") or href.removesuffix(".vcf")
    name = fm.get("name", "")
    firstname = fm.get("firstname", "")
    lastname = fm.get("lastname", "")
    org = fm.get("organization", "")
    job_title = fm.get("job_title", "")
    birthday = fm.get("birthday", "")
    address = fm.get("address", "")
    url = fm.get("url", "")

    emails = fm.get("email", [])
    if isinstance(emails, str):
        emails = [emails]
    phones = fm.get("phone", [])
    if isinstance(phones, str):
        phones = [phones]

    vc = vobject.vCard()
    vc.add("uid").value = uid
    vc.add("fn").value = name or f"{firstname} {lastname}".strip() or uid
    n = vobject.vcard.Name(family=lastname, given=firstname)
    vc.add("n").value = n
    if org:
        vc.add("org").value = [org]
    if job_title:
        vc.add("title").value = job_title
    for email in emails:
        e = vc.add("email")
        e.value = email
        e.type_param = "INTERNET"
    for phone in phones:
        p = vc.add("tel")
        p.value = str(phone)
    if birthday:
        vc.add("bday").value = str(birthday).replace("-", "")
    if address:
        adr = vc.add("adr")
        adr.value = vobject.vcard.Address(street=address)
    if url:
        vc.add("url").value = url
    if notes:
        vc.add("note").value = notes

    return vc.serialize()


def _vcf_to_md(vcf_text: str) -> Optional[Tuple[str, dict]]:
    """Parse a vCard string → (md_content, meta_dict)."""
    try:
        vc = vobject.readOne(vcf_text)
    except Exception:
        return None

    uid = str(vc.uid.value) if hasattr(vc, "uid") else str(uuid.uuid4())
    name = str(vc.fn.value) if hasattr(vc, "fn") else ""

    firstname, lastname = "", ""
    if hasattr(vc, "n"):
        n = vc.n.value
        firstname = str(n.given or "")
        lastname = str(n.family or "")

    emails = [str(e.value) for e in vc.contents.get("email", [])]
    phones = [str(p.value) for p in vc.contents.get("tel", [])]
    org = ""
    if hasattr(vc, "org"):
        org_val = vc.org.value
        org = org_val[0] if isinstance(org_val, list) else str(org_val)
    job_title = str(vc.title.value) if hasattr(vc, "title") else ""
    birthday = ""
    if hasattr(vc, "bday"):
        raw = str(vc.bday.value)
        if len(raw) == 8 and raw.isdigit():
            birthday = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        else:
            birthday = raw
    address = ""
    if hasattr(vc, "adr"):
        adr = vc.adr.value
        parts_adr = [adr.street, adr.city, adr.region, adr.code, adr.country]
        address = ", ".join(p for p in parts_adr if p)
    url = str(vc.url.value) if hasattr(vc, "url") else ""
    notes = str(vc.note.value) if hasattr(vc, "note") else ""

    fm: dict = {"uid": uid, "name": name}
    if firstname:
        fm["firstname"] = firstname
    if lastname:
        fm["lastname"] = lastname
    if emails:
        fm["email"] = emails[0] if len(emails) == 1 else emails
    if phones:
        fm["phone"] = phones[0] if len(phones) == 1 else [str(p) for p in phones]
    if org:
        fm["organization"] = org
    if job_title:
        fm["job_title"] = job_title
    if birthday:
        fm["birthday"] = birthday
    if address:
        fm["address"] = address
    if url:
        fm["url"] = url

    content = f"---\n{yaml.dump(fm, allow_unicode=True, default_flow_style=False).rstrip()}\n---\n"
    if notes:
        content += f"\n{notes}\n"

    meta = {"name": name, "uid": uid}
    return content, meta


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class Collection(BaseCollection):
    """A calendar or contacts collection backed by picoclaw session files."""

    def __init__(self, storage: "Storage", path: str, login: str, ctype: str) -> None:
        self._storage = storage
        self._path = path.strip("/")
        self._login = login          # the URL path component / htpasswd username
        self._user = _resolve_user_id(login)  # picoclaw user_id (may differ)
        self._ctype = ctype  # "calendar" or "contacts"

    # ── BaseCollection required properties ──

    @property
    def path(self) -> str:
        return self._path

    @property
    def tag(self) -> str:
        return "VCALENDAR" if self._ctype == "calendar" else "VADDRESSBOOK"

    # ── Filesystem helpers ──

    def _data_dir(self) -> str:
        base = self._storage._session_dir
        if self._ctype == "calendar":
            d = os.path.join(base, self._user, "workspace", "calendar")
        else:
            d = os.path.join(base, self._user, "workspace", "contacts")
        os.makedirs(d, exist_ok=True)
        return d

    def _href_to_filepath(self, href: str) -> Optional[str]:
        """Resolve a CalDAV/CardDAV href to an absolute .md file path, or None."""
        d = self._data_dir()
        if self._ctype == "calendar":
            md_name = href.removesuffix(".ics") + ".md"
        else:
            md_name = href.removesuffix(".vcf") + ".md"
        fp = os.path.join(d, md_name)
        return fp if os.path.exists(fp) else None

    def _iter_hrefs(self) -> Iterator[str]:
        d = self._data_dir()
        try:
            names = os.listdir(d)
        except FileNotFoundError:
            return
        ext = ".ics" if self._ctype == "calendar" else ".vcf"
        for f in names:
            if f.endswith(".md"):
                yield f.removesuffix(".md") + ext

    def _build_item(self, href: str, filepath: str) -> Optional[radicale_item.Item]:
        if self._ctype == "calendar":
            text = _md_to_ical(filepath, href)
        else:
            text = _md_to_vcf(filepath, href)

        if not text:
            return None

        return radicale_item.Item(
            collection_path=self._path,
            collection=self,
            href=href,
            last_modified=_mtime_rfc7231(filepath),
            text=text,
            etag=_etag(text),
        )

    # ── BaseCollection interface ──

    def get_multi(self, hrefs: Iterable[str]) -> Iterable[Tuple[str, Optional[radicale_item.Item]]]:
        for href in hrefs:
            fp = self._href_to_filepath(href)
            yield href, (self._build_item(href, fp) if fp else None)

    def get_all(self) -> Iterable[radicale_item.Item]:
        d = self._data_dir()
        for href in self._iter_hrefs():
            fp = self._href_to_filepath(href)
            if fp:
                item = self._build_item(href, fp)
                if item:
                    yield item

    def upload(self, href: str, item: radicale_item.Item) -> radicale_item.Item:
        d = self._data_dir()
        text = item.serialize

        if self._ctype == "calendar":
            result = _ical_to_md(text)
            if result is None:
                raise ValueError(f"Failed to parse iCal for href={href}")
            md_content, meta = result

            # Canonical filename: YYYY-MM-DD Safe-Title.md
            safe_title = _safe_filename(meta["title"])
            md_name = f"{meta['date_str']} {safe_title}.md" if safe_title else href.removesuffix(".ics") + ".md"

            # Avoid collision with different events on same date+title
            filepath = os.path.join(d, md_name)
            if os.path.exists(filepath):
                # Check if it's the same event (same uid in frontmatter)
                existing_href = md_name.removesuffix(".md") + ".ics"
                fp2 = self._href_to_filepath(existing_href)
                if fp2:
                    # Overwrite only if uid matches
                    try:
                        with open(fp2, encoding="utf-8") as f:
                            raw = f.read()
                        fm_existing = yaml.safe_load(raw.split("---", 2)[1]) or {}
                        if fm_existing.get("uid") != meta["uid"]:
                            # Different event — append index to avoid overwrite
                            n = 2
                            while os.path.exists(os.path.join(d, f"{meta['date_str']} {safe_title} {n}.md")):
                                n += 1
                            md_name = f"{meta['date_str']} {safe_title} {n}.md"
                            filepath = os.path.join(d, md_name)
                    except Exception:
                        pass

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

            canonical_href = md_name.removesuffix(".md") + ".ics"
        else:
            # Contacts: parse vCard → write as .md
            result = _vcf_to_md(text)
            if result is None:
                raise ValueError(f"Failed to parse vCard for href={href}")
            md_content, meta = result

            safe_name = _safe_filename(meta["name"]) if meta["name"] else href.removesuffix(".vcf")
            md_name = f"{safe_name}.md" if safe_name else href.removesuffix(".vcf") + ".md"
            filepath = os.path.join(d, md_name)

            # Collision check: overwrite only if same uid
            if os.path.exists(filepath):
                try:
                    with open(filepath, encoding="utf-8") as f:
                        raw = f.read()
                    fm_ex = yaml.safe_load(raw.split("---", 2)[1]) or {}
                    if fm_ex.get("uid") != meta["uid"]:
                        n = 2
                        while os.path.exists(os.path.join(d, f"{safe_name} {n}.md")):
                            n += 1
                        md_name = f"{safe_name} {n}.md"
                        filepath = os.path.join(d, md_name)
                except Exception:
                    pass

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
            canonical_href = md_name.removesuffix(".md") + ".vcf"

        stored = self._build_item(canonical_href, filepath)
        if stored is None:
            raise ValueError(f"Failed to build item after upload: {canonical_href}")
        return stored

    def delete(self, href: Optional[str] = None) -> None:
        if href is None:
            import shutil
            d = self._data_dir()
            if os.path.exists(d):
                shutil.rmtree(d)
            return
        fp = self._href_to_filepath(href)
        if fp and os.path.exists(fp):
            os.remove(fp)

    def get_meta(self, key: Optional[str] = None) -> Mapping[str, str]:
        props: dict = {
            "D:displayname": "Kalender" if self._ctype == "calendar" else "Kontakte",
            "CS:getctag": self.etag,
        }
        if self._ctype == "calendar":
            props["C:supported-calendar-component-set"] = "VEVENT"
        if key:
            return props.get(key)  # type: ignore[return-value]
        return props

    def set_meta(self, props: Mapping[str, str]) -> None:
        pass  # collection metadata not persisted

    @property
    def last_modified(self) -> str:
        d = self._data_dir()
        try:
            ts = max(
                (os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)),
                default=0.0,
            )
        except Exception:
            ts = 0.0
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    @property
    def etag(self) -> str:
        d = self._data_dir()
        try:
            ts = max(
                (os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)),
                default=0.0,
            )
        except Exception:
            ts = 0.0
        return _etag(str(ts))

    @property
    def serialize(self) -> str:
        if self._ctype == "calendar":
            cal = Calendar()
            cal.add("prodid", "-//Picoclaw//Picoclaw CalDAV//EN")
            cal.add("version", "2.0")
            for item in self.get_all():
                try:
                    sub = Calendar.from_ical(item.serialize)
                    for comp in sub.walk():
                        if comp.name == "VEVENT":
                            cal.add_component(comp)
                except Exception:
                    pass
            return cal.to_ical().decode("utf-8")
        else:
            return "\n".join(item.serialize for item in self.get_all())


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class Storage(BaseStorage):
    """Picoclaw Radicale storage — maps CalDAV/CardDAV to session workspace files."""

    def __init__(self, configuration) -> None:
        super().__init__(configuration)
        self._session_dir = os.environ.get("PICOCLAW_SESSION_DIR", "/session")
        self._lock = threading.Lock()

    # ── Path parsing ──

    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """(user, ctype, href) — any may be None if absent."""
        parts = [p for p in path.strip("/").split("/") if p]
        if not parts:
            return None, None, None
        user = parts[0]
        if len(parts) == 1:
            return user, None, None
        ctype = parts[1] if parts[1] in ("calendar", "contacts") else None
        href = parts[2] if ctype and len(parts) >= 3 else None
        return user, ctype, href

    def _user_exists(self, login: str) -> bool:
        user_id = _resolve_user_id(login)
        return os.path.isdir(os.path.join(self._session_dir, user_id))

    # ── BaseStorage interface ──

    @contextmanager
    def acquire_lock(self, mode: str, user: str = "") -> Iterator[None]:
        with self._lock:
            yield

    def discover(self, path: str, depth: str = "0") -> Iterable:
        login, ctype, href = self._parse_path(path)

        # Root "/" — yield nothing (no global listing)
        if login is None:
            return

        if not self._user_exists(login):
            return

        if ctype is None:
            # Principal path: yield both collections
            for ct in ("calendar", "contacts"):
                col = Collection(self, f"{login}/{ct}", login, ct)
                yield col
                if depth == "1":
                    yield from col.get_all()
            return

        col = Collection(self, f"{login}/{ctype}", login, ctype)

        if href is None:
            # Collection path
            yield col
            if depth == "1":
                yield from col.get_all()
        else:
            # Single item
            fp = col._href_to_filepath(href)
            if fp:
                item = col._build_item(href, fp)
                if item:
                    yield item

    def move(self, item: radicale_item.Item, to_collection: Collection, to_href: str) -> None:
        stored = to_collection.upload(to_href, item)
        item.collection.delete(item.href)

    def create_collection(self, path: str, items=None, props=None) -> Collection:
        login, ctype, _ = self._parse_path(path)
        if not login or not ctype:
            raise ValueError(f"Cannot create collection at path: {path}")
        col = Collection(self, f"{login}/{ctype}", login, ctype)
        col._data_dir()  # ensure directory exists
        if items:
            for item_href, item in items:
                col.upload(item_href, item)
        return col
