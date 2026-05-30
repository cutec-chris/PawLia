"""Discord interface for PawLia using discord.py.

Config (in config.yaml under "interfaces.discord"):

    discord:
      token: YOUR_BOT_TOKEN
      # allowed_users:           # optional — restrict to these user IDs
      #   - "123456789"
      # allowed_channels:        # optional — restrict to these channel IDs
      #   - "987654321"
      # always_thread: true      # auto-create threads on every message (default: true)
      # require_mention: false   # only respond to @mention (default: false)
      # slash_commands: true     # register slash commands (default: true)
"""

import asyncio
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Any

import discord
from discord.ext import commands

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.discord")

_DISCORD_MAX_MESSAGE_LENGTH = 2000
_DISCORD_MAX_EMBED_DESC = 4096

_AUDIO_FILE_EXTENSIONS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".pcm", ".webm", ".aac"}
_IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
_TEXT_FILE_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm", ".log",
    ".py", ".js", ".ts", ".rs", ".go", ".c", ".h", ".cpp", ".java", ".sh", ".bash",
}
_OFFICE_FILE_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".rtf", ".odt", ".ods", ".odp",
}


def _clean_discord_id(entry: str) -> str:
    entry = entry.strip()
    if entry.startswith("<@") and entry.endswith(">"):
        entry = entry.lstrip("<@!").rstrip(">")
    return entry


def _resolve_thread_id(message: discord.Message) -> Optional[int]:
    if isinstance(message.channel, discord.Thread):
        return message.channel.id
    return None


def _cmd(text: str, command: str) -> Optional[str]:
    for prefix in (f"//{command}", f"/{command}"):
        if text == prefix:
            return ""
        if text.startswith(prefix) and text[len(prefix)] in (" ", "\t", "\n"):
            return text[len(prefix):].strip()
    return None


async def start_discord(app: "App", cfg: Dict) -> None:
    """Connect to Discord and start handling messages.

    ``cfg`` is the ``interfaces.discord`` section of config.yaml.
    """
    token: str = cfg["token"]
    if not token:
        logger.error("Discord: no token configured")
        return

    allowed_users: Optional[List[str]] = cfg.get("allowed_users")
    allowed_channels: Optional[List[str]] = cfg.get("allowed_channels")
    always_thread: bool = cfg.get("always_thread", True)
    require_mention: bool = cfg.get("require_mention", False)
    slash_commands_enabled: bool = cfg.get("slash_commands", True)
    voice_cfg: Dict = cfg.get("voice", {})

    allowed_user_ids: Set[str] = set()
    if allowed_users:
        allowed_user_ids = {_clean_discord_id(uid) for uid in allowed_users}
    allowed_channel_ids: Set[str] = set()
    if allowed_channels:
        allowed_channel_ids = {_clean_discord_id(cid) for cid in allowed_channels}

    # ------------------------------------------------------------------
    # Opus codec loading
    # ------------------------------------------------------------------
    if not discord.opus.is_loaded():
        import ctypes.util
        opus_path = ctypes.util.find_library("opus")
        if not opus_path and hasattr(os, "sysconf") and os.sysconf_names.get("SC_PAGE_SIZE"):
            for path in (
                "/usr/lib/libopus.so",
                "/usr/lib/x86_64-linux-gnu/libopus.so",
                "/usr/lib/aarch64-linux-gnu/libopus.so",
            ):
                if os.path.isfile(path):
                    opus_path = path
                    break
        if opus_path:
            try:
                discord.opus.load_opus(opus_path)
            except Exception:
                logger.warning("Discord: opus found at %s but failed to load", opus_path)
        if not discord.opus.is_loaded():
            logger.warning("Discord: opus codec not found — voice channel support disabled")

    # ------------------------------------------------------------------
    # Agent cache (same pattern as Matrix)
    # ------------------------------------------------------------------
    from pawlia.interfaces.common import (
        AgentCache, build_status, format_status, handle_model_command,
        list_available_models, preview_text, format_private_toggle,
        format_bg_enqueue, bytes_to_data_uri, handle_reload_command,
    )

    agent_cache = AgentCache(app)

    def _session_id(guild_id: int, channel_id: int) -> str:
        return f"dc_{guild_id}_{channel_id}"

    def get_agent(guild_id: int, channel_id: int, thread_id: Optional[int] = None) -> Any:
        sid = _session_id(guild_id, channel_id)
        key = f"{sid}#{thread_id}" if thread_id else sid
        return agent_cache.get(sid, cache_key=key)

    # ------------------------------------------------------------------
    # Bot setup
    # ------------------------------------------------------------------
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guild_messages = True
    intents.dm_messages = True
    intents.voice_states = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    # ------------------------------------------------------------------
    # Voice manager
    # ------------------------------------------------------------------
    from pawlia.interfaces.discord_voice import VoiceManager

    voice_manager = VoiceManager(
        app=app,
        bot=bot,
        cfg=voice_cfg,
        allowed_user_ids=allowed_user_ids,
        get_agent_cb=get_agent,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_allowed(message: discord.Message) -> bool:
        if message.author == bot.user:
            return False
        if message.author.bot:
            return False
        if allowed_user_ids:
            if str(message.author.id) not in allowed_user_ids:
                return False
        if allowed_channel_ids:
            channel_id = str(message.channel.id)
            parent_id = None
            if isinstance(message.channel, discord.Thread) and message.channel.parent_id:
                parent_id = str(message.channel.parent_id)
            check_ids = {channel_id}
            if parent_id:
                check_ids.add(parent_id)
            if not (check_ids & allowed_channel_ids):
                return False
        return True

    async def _send(channel: discord.abc.Messageable, text: str) -> Optional[discord.Message]:
        if len(text) > _DISCORD_MAX_MESSAGE_LENGTH:
            chunks = []
            remaining = text
            while len(remaining) > _DISCORD_MAX_MESSAGE_LENGTH:
                split_at = remaining.rfind("\n", 0, _DISCORD_MAX_MESSAGE_LENGTH)
                if split_at < 100:
                    split_at = remaining.rfind(" ", 0, _DISCORD_MAX_MESSAGE_LENGTH)
                if split_at < 100:
                    split_at = _DISCORD_MAX_MESSAGE_LENGTH
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at:].lstrip()
            chunks.append(remaining)
            last_msg = None
            for chunk in chunks:
                last_msg = await channel.send(chunk)
            return last_msg
        return await channel.send(text)

    async def _reply(message: discord.Message, text: str) -> Optional[discord.Message]:
        try:
            if len(text) > _DISCORD_MAX_MESSAGE_LENGTH:
                # For long messages, send first as reply, rest as follow-ups
                first_chunk = text[:_DISCORD_MAX_MESSAGE_LENGTH]
                split_at = first_chunk.rfind("\n", 0, _DISCORD_MAX_MESSAGE_LENGTH)
                if split_at < 100:
                    split_at = first_chunk.rfind(" ", 0, _DISCORD_MAX_MESSAGE_LENGTH)
                if split_at < 100:
                    split_at = _DISCORD_MAX_MESSAGE_LENGTH
                first_chunk = text[:split_at]
                last_msg = await message.reply(first_chunk)
                remaining = text[split_at:].lstrip()
                while len(remaining) > _DISCORD_MAX_MESSAGE_LENGTH:
                    split_at = remaining.rfind("\n", 0, _DISCORD_MAX_MESSAGE_LENGTH)
                    if split_at < 100:
                        split_at = remaining.rfind(" ", 0, _DISCORD_MAX_MESSAGE_LENGTH)
                    if split_at < 100:
                        split_at = _DISCORD_MAX_MESSAGE_LENGTH
                    last_msg = await message.channel.send(remaining[:split_at])
                    remaining = remaining[split_at:].lstrip()
                if remaining:
                    last_msg = await message.channel.send(remaining)
                return last_msg
            return await message.reply(text)
        except (discord.NotFound, discord.HTTPException):
            return await _send(message.channel, text)

    async def _send_thread_reply(
        message: discord.Message, thread: discord.Thread, text: str
    ) -> Optional[discord.Message]:
        try:
            return await thread.send(text)
        except (discord.NotFound, discord.HTTPException):
            return await _send(message.channel, text)

    # ------------------------------------------------------------------
    # File type helpers
    # ------------------------------------------------------------------

    def _file_ext(filename: str) -> str:
        return os.path.splitext(filename or "")[1].lower()

    async def _download_attachment(att: discord.Attachment) -> Optional[bytes]:
        try:
            return await att.read()
        except Exception as e:
            logger.warning("Discord: failed to download attachment %s: %s", att.filename, e)
            return None

    # ------------------------------------------------------------------
    # Voice message handling
    # ------------------------------------------------------------------

    async def _handle_voice_message(
        message: discord.Message,
        attachment: discord.Attachment,
    ) -> None:
        data = await _download_attachment(attachment)
        if not data:
            return

        await bot_typing_indicator(message.channel)

        from pawlia.transcription import transcribe

        session_id = _session_id(
            message.guild.id if message.guild else 0,
            message.channel.id,
        )
        session = app.memory.load_session(session_id)
        thread_id = _resolve_thread_id(message)
        active_model = app.llm.default_model_name(
            "chat",
            agent_overrides=app.memory.effective_agent_overrides(session, thread_id),
        ) if hasattr(app.llm, "default_model_name") else None

        audio_info = app.llm.audio_model_info(active_model or "chat") if active_model and hasattr(app.llm, "audio_model_info") else None
        if audio_info:
            from pawlia.transcription import transcribe_via_model
            mime = attachment.content_type or "audio/ogg"
            text = await transcribe_via_model(data, audio_info[0], audio_info[1], mime=mime)
        else:
            mime = attachment.content_type or "audio/ogg"
            text = await transcribe(data, app.config, mime=mime)

        if not text:
            await _reply(message, "*(Sprachnachricht konnte nicht transkribiert werden)*")
            return

        logger.info("Discord: voice message transcribed: %s", text[:120])
        # Show transcription and route to agent
        await _reply(message, f":microphone2: *{text}*")
        await _handle_agent_message(message, f"[Sprachnachricht]: {text}")

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    async def _handle_image_message(
        message: discord.Message,
        images: List[discord.Attachment],
    ) -> None:
        data_uris = []
        for att in images[:4]:  # max 4 images per message
            data = await _download_attachment(att)
            if data:
                mimetype = att.content_type or "image/png"
                data_uri = bytes_to_data_uri(data, mimetype)
                data_uris.append(data_uri)

        if not data_uris:
            return

        caption = message.content.strip() if message.content else ""
        await _handle_agent_message(message, caption, images=data_uris)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    async def _handle_file_message(
        message: discord.Message,
        attachment: discord.Attachment,
    ) -> None:
        data = await _download_attachment(attachment)
        if not data:
            return

        filename = attachment.filename or "attachment"
        ext = _file_ext(filename)

        if ext in _TEXT_FILE_EXTENSIONS:
            for encoding in ("utf-8-sig", "utf-8", "latin-1"):
                try:
                    text = data[: 128 * 1024].decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = data[: 128 * 1024].decode("utf-8", errors="replace")
            truncated = len(data) > 128 * 1024
            suffix = "\n\n[Hinweis: Dateiinhalt wurde wegen Groesse gekuerzt.]" if truncated else ""
            await _handle_agent_message(
                message,
                f"[Datei empfangen]\nName: {filename}\n\nInhalt:\n```\n{text}\n```{suffix}"
            )
            return

        if ext in _OFFICE_FILE_EXTENSIONS:
            try:
                from markitdown import MarkItDown
                suffix = os.path.splitext(filename)[1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(data)
                    temp_path = f.name
                try:
                    converter = MarkItDown()
                    convert_local = getattr(converter, "convert_local", None)
                    result = convert_local(temp_path) if convert_local else converter.convert(temp_path)
                    md_text = getattr(result, "text_content", None) or str(result)
                    md_text = md_text.strip()
                    if md_text:
                        await _handle_agent_message(
                            message,
                            f"[Datei empfangen]\nName: {filename}\n\nInhalt:\n```markdown\n{md_text[:30000]}\n```"
                        )
                        return
                finally:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
            except Exception as e:
                logger.warning("Discord: MarkItDown conversion failed for %s: %s", filename, e)

        await _handle_agent_message(
            message,
            f"[Datei empfangen]\nName: {filename}\nTyp: {attachment.content_type or 'unbekannt'}\nDie Datei wurde nicht inhaltlich gelesen."
        )

    # ------------------------------------------------------------------
    # Core message handler
    # ------------------------------------------------------------------

    async def bot_typing_indicator(channel: discord.abc.Messageable) -> None:
        try:
            async with channel.typing():
                pass
        except Exception:
            pass

    async def _handle_agent_message(
        message: discord.Message,
        text: str,
        images: Optional[List[str]] = None,
    ) -> None:
        if not text.strip() and not images:
            return

        guild_id = message.guild.id if message.guild else 0
        channel_id = message.channel.id
        thread_id = _resolve_thread_id(message)
        session_id = _session_id(guild_id, channel_id)

        app.scheduler.touch_activity(session_id)

        # --- Slash commands ---
        if _cmd(text, "status") is not None:
            agent = get_agent(guild_id, channel_id, thread_id)
            status = build_status(app, session_id, agent, thread_id=thread_id)
            text_out = format_status(status)
            await _reply(message, text_out)
            return

        if _cmd(text, "private") is not None:
            if not thread_id:
                await _reply(message, "_//private funktioniert nur in Threads._")
                return
            session = app.memory.load_session(session_id)
            active = app.memory.toggle_private_thread(session, str(thread_id))
            await _reply(message, format_private_toggle(active))
            return

        if _cmd(text, "reload") is not None:
            result = handle_reload_command(app)
            agent_cache.invalidate_all()
            logger.info("Discord: app config reloaded")
            await _reply(message, result.message)
            return

        model_args = _cmd(text, "model")
        if model_args is not None:
            ctx_label = f"Thread {str(thread_id)[:8]}" if thread_id else "Channel"
            result = handle_model_command(app, session_id, model_args, thread_id=thread_id, ctx_label=ctx_label)
            if result.invalidate_agent:
                agent_cache.invalidate(session_id)
                logger.info("Discord: model changed for %s -> %s", session_id, result.model)
            elif result.action == "set":
                logger.info(
                    "Discord: model changed for %s thread %s -> %s",
                    session_id, str(thread_id)[:8] if thread_id else "", result.model,
                )

            avail = ", ".join(f"`{m}`" for m in result.available) or "_(keine konfiguriert)_"
            if result.action == "show" and result.chains:
                from pawlia.interfaces.common import format_model_chains
                chain_text = format_model_chains(result.chains)
                await _reply(
                    message,
                    f"{chain_text}\n\n"
                    f"**Verfügbar:** {avail}\n"
                    f"_Session-Chatmodell setzen: `//model <modell>` — Agent setzen: `//model <pfad> <modell>` — Löschen: `//model <pfad> off`_"
                )
            elif result.action == "show":
                await _reply(
                    message,
                    f"**Aktives Chat-Modell** [{result.ctx_label}]: `{result.model}`\n"
                    f"**Verfügbar:** {avail}\n"
                    f"_Session-Chatmodell setzen: `//model <modell>` — Agent setzen: `//model <pfad> <modell>` — Löschen: `//model <pfad> off`_"
                )
            elif result.action == "invalid_path":
                await _reply(message, "Ungültiger Model-Pfad. Erlaubt: `default`, `chat`, `skill_runner`, `vision`, `compiler`, `skills.<name>`.")
            elif result.action == "cleared":
                await _reply(message, f"✓ Model-Override `{result.path}` für **{result.ctx_label}** entfernt.")
            else:
                await _reply(message, f"✓ Model-Override `{result.path}` für **{result.ctx_label}** auf `{result.model}` gesetzt.")
            return

        if _cmd(text, "clear") is not None:
            if not thread_id:
                await _reply(message, "_//clear funktioniert nur in Threads._")
                return
            try:
                thread = bot.get_channel(thread_id)
                if thread and isinstance(thread, discord.Thread):
                    async for msg in thread.history(limit=100):
                        if msg.author == bot.user:
                            try:
                                await msg.delete()
                            except Exception:
                                pass
                    await _reply(message, "_Thread-Nachrichten gelöscht._")
                else:
                    await _reply(message, "_Thread nicht gefunden._")
            except Exception as e:
                logger.warning("Discord: clear failed: %s", e)
                await _reply(message, f"_Löschen fehlgeschlagen: {e}_")
            return

        bg_args = _cmd(text, "background")
        if bg_args is not None:
            if not bg_args:
                await _reply(message, "_Verwendung: //background <Nachricht>_")
                return
            app.scheduler.bg_tasks.enqueue(session_id, bg_args)
            await _reply(message, format_bg_enqueue(bg_args))
            return

        # --- Thread command ---
        thread_args = _cmd(text, "thread")
        if thread_args is not None:
            if not thread_args:
                await _reply(message, "_Verwendung: //thread <Nachricht>_")
                return
            try:
                thread_name = thread_args[:80]
                if isinstance(message.channel, discord.TextChannel):
                    new_thread = await message.create_thread(name=thread_name)
                    await new_thread.send(f"**Thread:** {thread_args}")
                    text = f"{thread_args}"
                    # Recurse into the thread
                    guild_id = message.guild.id if message.guild else 0
                    session_id = _session_id(guild_id, new_thread.id)
                    agent = get_agent(guild_id, new_thread.id, None)
                    response = await agent.run(text)
                    await new_thread.send(response)
                    return
                else:
                    await _reply(message, "_Threads sind nur in Server-Textkanälen verfügbar._")
                    return
            except Exception as e:
                logger.warning("Discord: thread creation failed: %s", e)
                await _reply(message, f"_Thread-Erstellung fehlgeschlagen: {e}_")
                return

        # --- Agent dispatch ---
        ctx = f" [thread {str(thread_id)[:8]}]" if thread_id else ""
        logger.info(
            "Discord: message in %s/%s%s: %s (images=%d)",
            guild_id, channel_id, ctx, text[:80], len(images or []),
        )

        await bot_typing_indicator(message.channel)

        agent = get_agent(guild_id, channel_id, thread_id)

        status_msg: Optional[discord.Message] = None
        step_count = 0
        current_skill: Optional[str] = None
        initial_query: Optional[str] = None

        async def _on_interim(interim_text: str) -> None:
            await _reply(message, interim_text)

        async def _on_skill_start(skill_name: str, query: str) -> None:
            nonlocal status_msg, step_count, current_skill, initial_query
            current_skill = skill_name
            initial_query = query
            step_count = 0
            short_q = (query[:60] + "…") if len(query) > 60 else query
            status_msg = await _reply(message, f"⚙ **{skill_name}**: {short_q}")

        async def _on_skill_step(step_text: str) -> None:
            nonlocal step_count
            step_count += 1
            short = (step_text[:100] + "…") if len(step_text) > 100 else step_text
            short_q = (initial_query[:60] + "…") if initial_query and len(initial_query) > 60 else (initial_query or "")
            if status_msg:
                try:
                    await status_msg.edit(content=f"⚙ **{current_skill}**: {short_q}\n[{step_count}] `{short}`")
                except (discord.NotFound, discord.HTTPException):
                    pass

        async def _on_skill_done(skill_name: str, result: str = "") -> None:
            if status_msg:
                short_q = (initial_query[:60] + "…") if initial_query and len(initial_query) > 60 else (initial_query or "")
                summary = ""
                if result:
                    clean = result.lstrip()
                    if clean.startswith("[Report from"):
                        for line in clean.splitlines():
                            if line.strip() and not line.startswith("[") and not line.startswith("---"):
                                clean = line
                                break
                    first_line = clean.splitlines()[0].strip() if clean.splitlines() else ""
                    summary_text = first_line if first_line else clean[:120]
                    summary = (summary_text[:120] + "…") if len(summary_text) > 120 else summary_text
                text = f"✓ **{skill_name}**: {short_q}\n({step_count} Schritte)"
                if summary:
                    text += f" — {summary}"
                try:
                    await status_msg.edit(content=text)
                except (discord.NotFound, discord.HTTPException):
                    pass

        agent.on_interim = _on_interim
        try:
            response = await agent.run(
                text, images=images or None, thread_id=str(thread_id) if thread_id else None,
                on_skill_start=_on_skill_start,
                on_skill_step=_on_skill_step,
                on_skill_done=_on_skill_done,
            )
        except Exception as e:
            logger.error("Discord: error processing message: %s", e)
            session = app.memory.load_session(session_id)
            override = app.memory.get_agent_override_value(session, "chat", thread_id=str(thread_id) if thread_id else None)
            hint = ""
            if override:
                avail = ", ".join(f"`{m}`" for m in list_available_models(app))
                hint = (
                    f"\n\n_Aktiver Model-Override: `{override}`. "
                    f"Wechseln mit `//model chat <modell>` oder löschen mit `//model chat off`._"
                    + (f"\n_Verfügbar: {avail}_" if avail else "")
                )
            await _reply(message, f"Fehler: {e}{hint}")
            return

        logger.info("Discord: response in %s/%s%s: %s", guild_id, channel_id, ctx, preview_text(response))
        await _reply(message, response)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @bot.event
    async def on_ready():
        logger.info("Discord: connected as %s (ID: %d)", bot.user, bot.user.id)

        if slash_commands_enabled:
            await _register_slash_commands()

    @bot.event
    async def on_message(message: discord.Message):
        if not _is_allowed(message):
            return

        # Skip empty messages without attachments
        if not message.content.strip() and not message.attachments:
            return

        # Require mention check
        if require_mention and not isinstance(message.channel, discord.DMChannel):
            if bot.user not in message.mentions and not message.guild:
                pass  # DM always passes
            elif bot.user not in message.mentions and message.guild:
                return

        # Handle attachments
        if message.attachments:
            for att in message.attachments:
                ext = _file_ext(att.filename)
                content_type = att.content_type or ""
                is_voice = getattr(att, "is_voice_message", False)

                if is_voice or (ext in _AUDIO_FILE_EXTENSIONS and "audio" in content_type):
                    await _handle_voice_message(message, att)
                    return
                elif "image" in content_type or ext in _IMAGE_FILE_EXTENSIONS:
                    image_atts = [
                        a for a in message.attachments
                        if ("image" in (a.content_type or "")) or _file_ext(a.filename) in _IMAGE_FILE_EXTENSIONS
                    ]
                    if image_atts:
                        await _handle_image_message(message, image_atts)
                    non_image_atts = [a for a in message.attachments if a not in image_atts]
                    for a in non_image_atts:
                        if not ("audio" in (a.content_type or "") or _file_ext(a.filename) in _AUDIO_FILE_EXTENSIONS):
                            await _handle_file_message(message, a)
                    return
                else:
                    await _handle_file_message(message, att)
                    return

        # Text message
        text = message.content.strip()
        if not text:
            return

        await _handle_agent_message(message, text)

    @bot.event
    async def on_voice_state_update(
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ):
        if member == bot.user:
            return
        if before.channel == after.channel:
            return
        await voice_manager.on_voice_state_update(member, before, after)

    # ------------------------------------------------------------------
    # Slash command registration
    # ------------------------------------------------------------------

    async def _register_slash_commands():
        @bot.tree.command(name="status", description="Zeigt Session-Status und Modell-Info an")
        async def slash_status(interaction: discord.Interaction):
            if not interaction.guild:
                await interaction.response.send_message("Befehle sind nur in Server-Kanälen verfügbar.", ephemeral=True)
                return
            session_id = _session_id(interaction.guild.id, interaction.channel.id)
            thread_id = interaction.channel.id if isinstance(interaction.channel, discord.Thread) else None
            agent = get_agent(interaction.guild.id, interaction.channel.id, thread_id)
            status = build_status(app, session_id, agent, thread_id=thread_id)
            await interaction.response.send_message(format_status(status))

        @bot.tree.command(name="model", description="Zeigt oder setzt das Chat-Modell")
        async def slash_model(interaction: discord.Interaction, modell: str = ""):
            if not interaction.guild:
                await interaction.response.send_message("Befehle sind nur in Server-Kanälen verfügbar.", ephemeral=True)
                return
            session_id = _session_id(interaction.guild.id, interaction.channel.id)
            thread_id = interaction.channel.id if isinstance(interaction.channel, discord.Thread) else None
            ctx_label = f"Thread {str(thread_id)[:8]}" if thread_id else "Channel"
            result = handle_model_command(app, session_id, modell, thread_id=thread_id, ctx_label=ctx_label)
            if result.invalidate_agent:
                agent_cache.invalidate(session_id)

            avail = ", ".join(f"`{m}`" for m in result.available) or "_(keine)_"
            if result.action == "show" and result.chains:
                from pawlia.interfaces.common import format_model_chains
                chain_text = format_model_chains(result.chains)
                await interaction.response.send_message(
                    f"{chain_text}\n\n**Verfügbar:** {avail}"
                )
            elif result.action == "show":
                await interaction.response.send_message(
                    f"**Aktives Modell** [{result.ctx_label}]: `{result.model}`\n**Verfügbar:** {avail}"
                )
            elif result.action == "cleared":
                await interaction.response.send_message(f"✓ Model-Override `{result.path}` entfernt.")
            elif result.action == "set":
                await interaction.response.send_message(f"✓ Model-Override `{result.path}` auf `{result.model}` gesetzt.")
            else:
                await interaction.response.send_message("Ungültiger Model-Pfad.")

        @bot.tree.command(name="background", description="Aufgabe im Hintergrund ausführen")
        async def slash_background(interaction: discord.Interaction, aufgabe: str):
            if not interaction.guild:
                await interaction.response.send_message("Befehle sind nur in Server-Kanälen verfügbar.", ephemeral=True)
                return
            session_id = _session_id(interaction.guild.id, interaction.channel.id)
            app.scheduler.bg_tasks.enqueue(session_id, aufgabe)
            await interaction.response.send_message(format_bg_enqueue(aufgabe))

        @bot.tree.command(name="reload", description="Konfiguration neu laden")
        async def slash_reload(interaction: discord.Interaction):
            result = handle_reload_command(app)
            agent_cache.invalidate_all()
            await interaction.response.send_message(result.message)

        @bot.tree.command(name="clear", description="Thread-Nachrichten des Bots löschen")
        async def slash_clear(interaction: discord.Interaction):
            if not isinstance(interaction.channel, discord.Thread):
                await interaction.response.send_message("Dieser Befehl funktioniert nur in Threads.", ephemeral=True)
                return
            try:
                count = 0
                async for msg in interaction.channel.history(limit=100):
                    if msg.author == bot.user:
                        try:
                            await msg.delete()
                            count += 1
                        except Exception:
                            pass
                await interaction.response.send_message(f"{count} Nachrichten gelöscht.", ephemeral=True)
            except Exception as e:
                await interaction.response.send_message(f"Löschen fehlgeschlagen: {e}", ephemeral=True)

        @bot.tree.command(name="voice", description="Voice-Channel steuern (join/leave)")
        async def slash_voice(interaction: discord.Interaction, aktion: str):
            if not interaction.guild:
                await interaction.response.send_message("Voice-Befehle sind nur in Servern verfügbar.", ephemeral=True)
                return
            if aktion == "join":
                if not interaction.user.voice or not interaction.user.voice.channel:
                    await interaction.response.send_message("Du bist in keinem Voice-Channel.", ephemeral=True)
                    return
                success = await voice_manager.join(interaction.user.voice.channel, interaction.channel)
                if success:
                    await interaction.response.send_message(f":microphone2: Voice-Channel **{interaction.user.voice.channel.name}** beigetreten.")
                else:
                    await interaction.response.send_message("Voice-Channel konnte nicht beigetreten werden.", ephemeral=True)
            elif aktion == "leave":
                await voice_manager.leave(interaction.guild.id)
                await interaction.response.send_message("Voice-Channel verlassen.")
            else:
                await interaction.response.send_message("Verwendung: `/voice join` oder `/voice leave`", ephemeral=True)

        try:
            await bot.tree.sync()
            logger.info("Discord: slash commands synced")
        except Exception as e:
            logger.warning("Discord: slash command sync failed: %s", e)

    # ------------------------------------------------------------------
    # Scheduler callback for proactive notifications
    # ------------------------------------------------------------------

    async def _discord_notify(session_id: str, message: str) -> None:
        # session_id for discord agents is "dc_{guild_id}_{channel_id}"
        if not session_id.startswith("dc_"):
            return
        parts = session_id[3:].rsplit("_", 1)
        if len(parts) != 2:
            return
        try:
            guild_id = int(parts[0])
            channel_id = int(parts[1])
        except ValueError:
            return
        channel = bot.get_channel(channel_id)
        if not channel:
            try:
                channel = await bot.fetch_channel(channel_id)
            except Exception:
                return
        if channel:
            msg = await _send(channel, message)
            if msg:
                session = app.memory.load_session(session_id)
                app.memory.seed_thread_context(session, str(msg.id), message)

    app.scheduler.register(_discord_notify)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    logger.info("Discord: starting bot...")
    try:
        await bot.start(token)
    except asyncio.CancelledError:
        logger.info("Discord: bot task cancelled")
    except Exception as e:
        logger.error("Discord: bot crashed: %s", e)
    finally:
        await voice_manager.cleanup_all()
        if not bot.is_closed():
            await bot.close()
        logger.info("Discord: disconnected")
