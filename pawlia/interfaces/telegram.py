"""Telegram interface for PawLia using python-telegram-bot.

Config (in config.json under "interfaces.telegram"):
    {
      "token": "YOUR_TELEGRAM_BOT_TOKEN"
    }
"""

import asyncio
import base64
import logging
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

    ``cfg`` is the ``interfaces.telegram`` section of config.json.
    """
    token: str = cfg["token"]

    from pawlia.interfaces.common import AgentCache, build_status, format_status, md_to_tg_html, handle_model_command

    # One agent per user; thread context is passed at run() time
    agent_cache = AgentCache(app)
    chat_ids: Dict[str, int] = {}

    async def _handle(update: Update, user_id: str, text: str,
                      thread_id: Optional[int] = None,
                      images: Optional[List[str]] = None) -> None:
        """Shared handler for text and photo messages."""
        try:
            # Show typing indicator while processing
            await update.message.chat.send_action(ChatAction.TYPING)

            agent = agent_cache.get(user_id)

            async def _on_interim(interim_text: str) -> None:
                await update.message.reply_text(
                    md_to_tg_html(interim_text), parse_mode=ParseMode.HTML,
                )
                # Re-send typing after interim message so it stays visible
                await update.message.chat.send_action(ChatAction.TYPING)

            status_message = None
            step_count = 0
            current_skill: Optional[str] = None

            async def _on_skill_start(skill_name: str, query: str) -> None:
                nonlocal status_message, step_count, current_skill
                current_skill = skill_name
                step_count = 0
                short_q = (query[:60] + "…") if len(query) > 60 else query
                status_message = await update.message.reply_text(
                    f"<i>⚙ {skill_name}: {short_q}</i>", parse_mode=ParseMode.HTML,
                )

            async def _on_skill_step(step_text: str) -> None:
                nonlocal step_count
                step_count += 1
                if status_message and current_skill:
                    short = (step_text[:100] + "…") if len(step_text) > 100 else step_text
                    try:
                        await status_message.edit_text(
                            f"<i>⚙ {current_skill} [{step_count}]: <code>{short}</code></i>",
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

            async def _on_skill_done(skill_name: str) -> None:
                if status_message:
                    try:
                        await status_message.edit_text(
                            f"<i>✓ {skill_name} ({step_count} Schritte)</i>",
                            parse_mode=ParseMode.HTML,
                        )
                    except Exception:
                        pass

            agent.on_interim = _on_interim
            agent.on_skill_start = _on_skill_start
            agent.on_skill_step = _on_skill_step
            agent.on_skill_done = _on_skill_done
            response = await agent.run(
                text,
                images=images or None,
                thread_id=str(thread_id) if thread_id else None,
            )
            await update.message.reply_text(
                md_to_tg_html(response), parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error("Telegram: error processing message: %s", e)
            await update.message.reply_text(f"Fehler: {e}")

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
        icon = "🔒" if active else "🔓"
        state = "aktiviert" if active else "deaktiviert"
        await update.message.reply_text(
            f"{icon} Private Mode {state} — Nachrichten werden <b>{'nicht ' if active else ''}gespeichert</b>.",
            parse_mode=ParseMode.HTML,
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
        """/model [name] — show or change the active model for this session."""
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

        if result.action == "show":
            await update.message.reply_text(
                f"<b>Aktives Modell</b> [{result.ctx_label}]: <code>{result.model}</code>",
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text(
                f"✓ Modell für <b>{result.ctx_label}</b> auf <code>{result.model}</code> gesetzt.",
                parse_mode=ParseMode.HTML,
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
        data = await file.download_as_bytearray()
        b64 = base64.b64encode(bytes(data)).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"

        caption = (update.message.caption or "").strip()

        logger.info("Telegram: photo from %s (%s), caption: %s", user.first_name, user_id, caption[:80])
        await _handle(update, user_id, caption, thread_id=thread_id, images=[data_uri])

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("private", on_private_command))
    application.add_handler(CommandHandler("model", on_model_command))
    application.add_handler(CommandHandler("thread", on_thread_command))
    application.add_handler(CommandHandler("status", on_status_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_message),
    )
    application.add_handler(
        MessageHandler(filters.PHOTO, on_photo),
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
