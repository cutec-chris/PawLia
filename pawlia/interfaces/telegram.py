"""Telegram interface for PawLia using python-telegram-bot.

Config (in config.json under "interfaces.telegram"):
    {
      "token": "YOUR_TELEGRAM_BOT_TOKEN"
    }
"""

import asyncio
import base64
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    filters,
)

if TYPE_CHECKING:
    from pawlia.app import App

logger = logging.getLogger("pawlia.interfaces.telegram")


def _md_to_tg_html(text: str) -> str:
    """Convert common markdown to Telegram-compatible HTML subset.

    Telegram supports: <b>, <i>, <code>, <pre>, <a href="">, <s>, <u>.
    """
    # Fenced code blocks: ```lang\ncode\n``` -> <pre>code</pre>
    text = re.sub(
        r"```(?:\w*)\n(.*?)```",
        lambda m: f"<pre>{m.group(1).rstrip()}</pre>",
        text,
        flags=re.DOTALL,
    )
    # Inline code: `code` -> <code>code</code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    # Links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


async def start_telegram(app: "App", cfg: Dict) -> None:
    """Start the Telegram bot and poll for messages.

    ``cfg`` is the ``interfaces.telegram`` section of config.json.
    """
    token: str = cfg["token"]

    # One agent per Telegram user, track chat_ids for proactive notifications
    agents: Dict[str, object] = {}
    chat_ids: Dict[str, int] = {}

    def get_agent(user_id: str):
        if user_id not in agents:
            agents[user_id] = app.make_agent(user_id)
        return agents[user_id]

    async def _handle(update: Update, user_id: str, text: str, images: Optional[List[str]] = None) -> None:
        """Shared handler for text and photo messages."""
        try:
            # Show typing indicator while processing
            await update.message.chat.send_action(ChatAction.TYPING)

            agent = get_agent(user_id)

            async def _on_interim(interim_text: str) -> None:
                await update.message.reply_text(
                    _md_to_tg_html(interim_text), parse_mode=ParseMode.HTML,
                )
                # Re-send typing after interim message so it stays visible
                await update.message.chat.send_action(ChatAction.TYPING)

            agent.on_interim = _on_interim
            response = await agent.run(text, images=images or None)
            await update.message.reply_text(
                _md_to_tg_html(response), parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            logger.error("Telegram: error processing message: %s", e)

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return

        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        chat_ids[user_id] = update.message.chat_id
        text = update.message.text.strip()
        if not text:
            return

        logger.info("Telegram: message from %s (%s): %s", user.first_name, user_id, text[:80])
        await _handle(update, user_id, text)

    async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.photo:
            return

        user = update.message.from_user
        if user is None:
            return

        user_id = f"tg_{user.id}"
        chat_ids[user_id] = update.message.chat_id

        # Grab the highest-resolution photo
        photo = update.message.photo[-1]
        file = await photo.get_file()
        data = await file.download_as_bytearray()
        b64 = base64.b64encode(bytes(data)).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"

        caption = (update.message.caption or "").strip()

        logger.info("Telegram: photo from %s (%s), caption: %s", user.first_name, user_id, caption[:80])
        await _handle(update, user_id, caption, images=[data_uri])

    application = Application.builder().token(token).build()
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
                    chat_id=chat_id, text=_md_to_tg_html(message), parse_mode=ParseMode.HTML,
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
