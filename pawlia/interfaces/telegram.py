"""Telegram interface for PawLia using python-telegram-bot.

Config (in config.yaml under "interfaces.telegram"):
    {
      "token": "YOUR_TELEGRAM_BOT_TOKEN"
    }
"""

import asyncio
import io
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.telegram")


async def start_telegram(app: "App", cfg: Dict) -> None:
    """Start the Telegram bot and poll for messages.

    ``cfg`` is the ``interfaces.telegram`` section of config.yaml.
    """
    token: str = cfg["token"]

    from pawlia import attachments
    from pawlia.interfaces.common import (
        AgentCache, build_status, format_status, md_to_tg_html,
        handle_model_command,
        list_available_models, preview_text,
        format_private_toggle, format_bg_enqueue, bytes_to_data_uri,
        handle_reload_command,
    )

    # One agent per user; thread context is passed at run() time
    agent_cache = AgentCache(app)
    chat_ids: Dict[str, int] = {}

    def _max_incoming_bytes() -> int:
        return int(
            (app.config.get("attachments") or {}).get("max_incoming_bytes")
            or 26214400
        )

    def _save_telegram_bytes(user_id: str, data: bytes, filename: str, mimetype: Optional[str]) -> None:
        try:
            attachments.save_incoming(
                session_dir=app.session_dir,
                user_id=user_id,
                data=data,
                filename=filename,
                source="telegram",
                mimetype=mimetype,
                max_bytes=_max_incoming_bytes(),
            )
        except Exception as exc:
            logger.warning("Telegram: failed to save incoming file %s: %s", filename, exc)

    async def _send_attachment(bot_inst, chat_id: int, thread_id: Optional[int], att: dict) -> None:
        data: bytes = att.get("data") or b""
        mimetype: str = att.get("mimetype") or "application/octet-stream"
        filename: str = att.get("filename") or "attachment"
        caption: Optional[str] = att.get("caption")
        kwargs = {"message_thread_id": thread_id} if thread_id else {}
        if caption:
            kwargs["caption"] = caption
        try:
            if mimetype.lower().startswith("image/"):
                await bot_inst.send_photo(
                    chat_id=chat_id, photo=data, filename=filename, **kwargs,
                )
            else:
                await bot_inst.send_document(
                    chat_id=chat_id, document=data, filename=filename, **kwargs,
                )
        except Exception as exc:
            logger.error("Telegram: failed to send attachment %s: %s", filename, exc)

    async def _handle(update: Update, user_id: str, text: str,
                      thread_id: Optional[int] = None,
                      images: Optional[List[str]] = None) -> None:
        """Shared handler for text and photo messages."""
        app.scheduler.touch_activity(user_id)
        try:
            await update.message.chat.send_action(ChatAction.TYPING)

            agent = agent_cache.get(user_id)

            async def _on_interim(interim_text: str) -> None:
                await update.message.reply_text(
                    md_to_tg_html(interim_text), parse_mode=ParseMode.HTML,
                )
                await update.message.chat.send_action(ChatAction.TYPING)

            status_message = None
            step_count = 0
            current_skill: Optional[str] = None
            initial_query: Optional[str] = None

            async def _on_skill_start(skill_name: str, query: str) -> None:
                nonlocal status_message, step_count, current_skill, initial_query
                current_skill = skill_name
                initial_query = query
                step_count = 0
                await update.message.chat.send_action(ChatAction.TYPING)
                short_q = (query[:60] + "…") if len(query) > 60 else query
                status_message = await update.message.reply_text(
                    f"<i>⚙ {skill_name}: {short_q}</i>", parse_mode=ParseMode.HTML,
                )

            async def _on_skill_step(step_text: str) -> None:
                nonlocal step_count
                step_count += 1
                await update.message.chat.send_action(ChatAction.TYPING)
                if status_message and current_skill:
                    short = (step_text[:100] + "…") if len(step_text) > 100 else step_text
                    try:
                        await status_message.edit_text(
                            f"<i>⚙ {current_skill} [{step_count}]: <code>{short}</code></i>",
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

            async def _on_skill_done(skill_name: str, result: str = "") -> None:
                await update.message.chat.send_action(ChatAction.TYPING)
                if status_message:
                    short_q = (initial_query[:60] + "…") if initial_query else skill_name
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
                    if summary:
                        text = f"<i>✓ {skill_name}: {short_q} ({step_count} Schritte) — {summary}</i>"
                    else:
                        text = f"<i>✓ {skill_name} ({step_count} Schritte)</i>"
                    try:
                        await status_message.edit_text(
                            text,
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

            agent.on_interim = _on_interim
            response = await agent.run(
                text,
                images=images or None,
                thread_id=str(thread_id) if thread_id else None,
                on_skill_start=_on_skill_start,
                on_skill_step=_on_skill_step,
                on_skill_done=_on_skill_done,
            )
            ctx_label = f" [thread {thread_id}]" if thread_id else ""
            logger.info("Telegram: response to %s%s: %s", user_id, ctx_label, preview_text(response))
            await update.message.reply_text(
                md_to_tg_html(response), parse_mode=ParseMode.HTML,
            )
            # Drain any attachments queued by direct tools (e.g. attach_file).
            bot_inst = update.get_bot()
            chat_id = update.message.chat_id
            for att in getattr(agent, "pending_attachments", []) or []:
                await _send_attachment(bot_inst, chat_id, thread_id, att)
        except Exception as e:
            logger.error("Telegram: error processing message: %s", e)
            session = app.memory.load_session(user_id)
            tid = str(thread_id) if thread_id else None
            override = app.memory.get_agent_override_value(session, "chat", thread_id=tid)
            hint = ""
            if override:
                avail = ", ".join(f"<code>{m}</code>" for m in list_available_models(app))
                hint = (
                    f"\n\n<i>Aktiver Model-Override: <code>{override}</code>. "
                    f"Wechseln mit /model &lt;name&gt; oder löschen mit /model off</i>"
                    + (f"\n<i>Verfügbar: {avail}</i>" if avail else "")
                )
            await update.message.reply_text(
                f"Fehler: {e}{hint}", parse_mode=ParseMode.HTML,
            )

    async def on_private_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/private — toggle private mode for the current thread (threads only)."""
        if not update.message:
            return
        user = update.message.from_user
        if user is None:
            return

        thread_id: Optional[int] = update.message.message_thread_id
        if not thread_id:
            await update.message.reply_text(
                "<i>/private funktioniert nur in Threads.</i>", parse_mode=ParseMode.HTML,
            )
            return

        user_id = f"tg_{user.id}"
        session = app.memory.load_session(user_id)
        active = app.memory.toggle_private_thread(session, str(thread_id))
        await update.message.reply_text(
            md_to_tg_html(format_private_toggle(active)), parse_mode=ParseMode.HTML,
        )

    async def on_thread_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/thread <message> — run a message in its own isolated thread context."""
        if not update.message:
            return
        user = update.message.from_user
        if user is None:
            return

        args = context.args or []
        if not args:
            await update.message.reply_text(
                "<i>Verwendung: /thread &lt;Nachricht&gt;</i>", parse_mode=ParseMode.HTML,
            )
            return

        user_id = f"tg_{user.id}"
        thread_id = str(update.message.message_id)
        text = " ".join(args)

        logger.info("Telegram: /thread from %s (%s): %s", user.first_name, user_id, text[:80])
        await _handle(update, user_id, text, thread_id=thread_id)

    async def on_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/status — show session status."""
        if not update.message:
            return
        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        thread_id: Optional[int] = update.message.message_thread_id
        agent = agent_cache.get(user_id)
        status = build_status(
            app, user_id, agent,
            thread_id=str(thread_id) if thread_id else None,
        )
        await update.message.reply_text(
            md_to_tg_html(format_status(status)), parse_mode=ParseMode.HTML,
        )

    async def on_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/model [model] or /model [path] [model] — show/change model selectors."""
        if not update.message:
            return
        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        thread_id: Optional[int] = update.message.message_thread_id
        args_str = " ".join(context.args) if context.args else ""

        result = handle_model_command(
            app, user_id, args_str,
            thread_id=str(thread_id) if thread_id else None,
        )

        if result.invalidate_agent:
            agent_cache.invalidate(user_id)
            logger.info("Telegram: model changed for %s -> %s", user_id, result.model)
        elif result.action == "set":
            logger.info("Telegram: model changed for %s thread %s -> %s", user_id, thread_id, result.model)

        avail = ", ".join(f"<code>{m}</code>" for m in result.available) or "<i>(keine konfiguriert)</i>"
        if result.action == "show" and result.chains:
            from pawlia.interfaces.common import format_model_chains
            chain_text = format_model_chains(result.chains)
            # Convert markdown bold/code to HTML for Telegram
            chain_text = chain_text.replace("**", "<b>").replace("**", "</b>")
            # Simple conversion: `code` → <code>code</code>
            import re
            chain_text = re.sub(r"`([^`]+)`", r"<code>\1</code>", chain_text)
            await update.message.reply_text(
                f"{chain_text}\n\n"
                f"<b>Verfügbar:</b> {avail}\n"
                f"<i>Session-Chatmodell setzen: /model &lt;modell&gt; — Agent setzen: /model &lt;pfad&gt; &lt;modell&gt; — Löschen: /model &lt;pfad&gt; off</i>",
                parse_mode=ParseMode.HTML,
            )
        elif result.action == "show":
            await update.message.reply_text(
                f"<b>Aktives Chat-Modell</b> [{result.ctx_label}]: <code>{result.model}</code>\n"
                f"<b>Verfügbar:</b> {avail}\n"
                f"<i>Session-Chatmodell setzen: /model &lt;modell&gt; — Agent setzen: /model &lt;pfad&gt; &lt;modell&gt; — Löschen: /model &lt;pfad&gt; off</i>",
                parse_mode=ParseMode.HTML,
            )
        elif result.action == "invalid_path":
            await update.message.reply_text(
                "Ungültiger Model-Pfad. Erlaubt: <code>default</code>, <code>chat</code>, <code>skill_runner</code>, <code>vision</code>, <code>compiler</code>, <code>skills.&lt;name&gt;</code>.",
                parse_mode=ParseMode.HTML,
            )
        elif result.action == "cleared":
            await update.message.reply_text(
                f"✓ Model-Override <code>{result.path}</code> für <b>{result.ctx_label}</b> entfernt.",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text(
                f"✓ Model-Override <code>{result.path}</code> für <b>{result.ctx_label}</b> auf <code>{result.model}</code> gesetzt.",
                parse_mode=ParseMode.HTML,
            )

    async def on_reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/reload — reload config-driven state and rebuild cached agents."""
        if not update.message:
            return
        result = handle_reload_command(app)
        agent_cache.invalidate_all()
        logger.info("Telegram: app config reloaded")
        await update.message.reply_text(
            md_to_tg_html(result.message), parse_mode=ParseMode.HTML,
        )

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return

        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        chat_ids[user_id] = update.message.chat_id
        thread_id: Optional[int] = update.message.message_thread_id
        text = update.message.text.strip()
        if not text:
            return

        ctx_label = f" [thread {thread_id}]" if thread_id else ""
        logger.info("Telegram: message from %s (%s)%s: %s", user.first_name, user_id, ctx_label, text[:80])
        await _handle(update, user_id, text, thread_id=thread_id)

    async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo:
            return

        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        chat_ids[user_id] = update.message.chat_id
        thread_id: Optional[int] = update.message.message_thread_id

        # Grab the highest-resolution photo
        photo = update.message.photo[-1]
        file = await photo.get_file()
        data = bytes(await file.download_as_bytearray())
        data_uri = bytes_to_data_uri(data, "image/jpeg")

        caption = (update.message.caption or "").strip()

        # Persist a copy under the user's Downloads/ folder so the user can
        # later ask the bot to re-send the photo via the attach_file tool.
        photo_filename = f"telegram-photo-{int.from_bytes(data[:8], 'big') & 0xffffff:06x}.jpg"
        _save_telegram_bytes(user_id, data, photo_filename, "image/jpeg")

        logger.info("Telegram: photo from %s (%s), caption: %s", user.first_name, user_id, caption[:80])
        await _handle(update, user_id, caption, thread_id=thread_id, images=[data_uri])

    async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle generic file uploads (PDFs, Office docs, archives, etc.)."""
        if not update.message or not update.message.document:
            return

        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        chat_ids[user_id] = update.message.chat_id
        thread_id: Optional[int] = update.message.message_thread_id

        doc = update.message.document
        try:
            tg_file = await doc.get_file()
            data = bytes(await tg_file.download_as_bytearray())
        except Exception as exc:
            logger.warning("Telegram: failed to download document: %s", exc)
            return

        filename = doc.file_name or "document"
        mimetype = doc.mime_type or "application/octet-stream"
        caption = (update.message.caption or "").strip()

        _save_telegram_bytes(user_id, data, filename, mimetype)

        logger.info(
            "Telegram: document from %s (%s): %s (%s, %d bytes)",
            user.first_name, user_id, filename, mimetype, len(data),
        )
        await _handle(update, user_id, caption, thread_id=thread_id)

    async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        voice = update.message.voice or update.message.audio
        if not voice:
            return
        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        chat_ids[user_id] = update.message.chat_id
        thread_id: Optional[int] = update.message.message_thread_id

        logger.info("Telegram: voice message from %s (%s)", user.first_name, user_id)

        try:
            file = await voice.get_file()
            data = bytes(await file.download_as_bytearray())
        except Exception as e:
            logger.warning("Telegram: failed to download audio: %s", e)
            await update.message.reply_text(
                "<i>(Sprachnachricht konnte nicht heruntergeladen werden)</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        from pawlia.transcription import transcribe

        # Resolve the active model for this user (respects session overrides)
        session = app.memory.load_session(user_id)
        active_model = app.llm.default_model_name(
            "chat",
            agent_overrides=app.memory.effective_agent_overrides(session),
        )
        audio_info = app.llm.audio_model_info(active_model or "chat")
        if audio_info:
            from pawlia.transcription import transcribe_via_model
            text = await transcribe_via_model(data, audio_info[0], audio_info[1], mime="audio/ogg")
        else:
            text = await transcribe(data, app.config, mime="audio/ogg")
        if not text:
            logger.warning("Telegram: transcription returned nothing")
            await update.message.reply_text(
                "<i>(Sprachnachricht konnte nicht transkribiert werden)</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        logger.info("Telegram: voice transcribed: %s", text[:120])
        # Show transcription in UI
        await update.message.reply_text(f"🎙️ {text}")
        # Route through normal handler (prefixed so agent knows it was voice)
        await _handle(update, user_id, f"[Sprachnachricht]: {text}", thread_id=thread_id)

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("private", on_private_command))
    application.add_handler(CommandHandler("model", on_model_command))
    application.add_handler(CommandHandler("reload", on_reload_command))
    application.add_handler(CommandHandler("thread", on_thread_command))
    application.add_handler(CommandHandler("status", on_status_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_message),
    )
    application.add_handler(
        MessageHandler(filters.PHOTO, on_photo),
    )
    application.add_handler(
        MessageHandler(filters.Document.ALL, on_document),
    )
    application.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, on_voice),
    )

    # Register scheduler callback for proactive notifications
    async def _tg_notify(user_id: str, message: str) -> None:
        chat_id = chat_ids.get(user_id)
        if chat_id:
            try:
                await application.bot.send_message(
                    chat_id=chat_id, text=md_to_tg_html(message), parse_mode=ParseMode.HTML,
                )
            except Exception as e:
                logger.error("Telegram notify failed for %s: %s", user_id, e)

    app.scheduler.register(_tg_notify)

    logger.info("Telegram: starting polling...")
    async with application:
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        try:
            await asyncio.Event().wait()  # run until cancelled
        except asyncio.CancelledError:
            pass
        finally:
            await application.updater.stop()
            await application.stop()
    logger.info("Telegram: disconnected")
