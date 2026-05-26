"""Voice channel handling for the Discord interface.

Provides VoiceManager that manages per-guild voice sessions:
- Joining/leaving voice channels
- Capturing user audio (opus decode, silence detection, PCM→WAV→STT)
- TTS playback into voice channel
"""

import asyncio
import logging
import os
import struct
import subprocess
import tempfile
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import discord

logger = logging.getLogger("pawlia.interfaces.discord_voice")

# Silence detection defaults
_SILENCE_THRESHOLD = 1.5   # seconds of silence → end of utterance
_MIN_SPEECH_DURATION = 0.5  # minimum seconds to process
_SAMPLE_RATE = 48000        # Discord native rate
_CHANNELS = 2               # Discord sends stereo
_PLAYBACK_TIMEOUT = 120     # max seconds to wait for playback
_VOICE_TIMEOUT = 300        # auto-disconnect after inactivity

if TYPE_CHECKING:
    from pawlia.app import App


class _VoiceReceiver:
    """Captures and decodes voice audio from a Discord voice channel.

    Attaches to a VoiceClient's socket listener, decrypts RTP packets,
    decodes Opus to PCM, and buffers per-user audio.  A polling loop
    detects silence and delivers completed utterances via a callback.
    """

    def __init__(
        self,
        voice_client: discord.VoiceClient,
        allowed_user_ids: Optional[Set[str]] = None,
    ):
        self._vc = voice_client
        self._allowed_user_ids = allowed_user_ids or set()
        self._running = False
        self._secret_key: Optional[bytes] = None
        self._bot_ssrc: int = 0
        self._ssrc_to_user: Dict[int, int] = {}
        self._buffers: Dict[int, bytearray] = defaultdict(bytearray)
        self._last_packet_time: Dict[int, float] = {}
        self._decoders: Dict[int, Any] = {}
        self._paused = False

    def start(self) -> None:
        if not self._vc or not hasattr(self._vc, "_connection"):
            return
        conn = self._vc._connection
        if self._vc.secret_key:
            self._secret_key = self._vc.secret_key
        if hasattr(self._vc, "user") and self._vc.user:
            self._bot_ssrc = getattr(self._vc, "_ssrc", 0)
        if hasattr(conn, "add_listener"):
            conn.add_listener(self._on_packet, "voice_data")
        self._running = True
        logger.info("Discord voice: receiver started for guild %d", self._vc.guild.id if self._vc.guild else 0)

    def stop(self) -> None:
        self._running = False
        if hasattr(self._vc, "_connection") and self._vc._connection:
            try:
                self._vc._connection.remove_listener(self._on_packet, "voice_data")
            except Exception:
                pass

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _on_packet(self, data: bytes) -> None:
        if not self._running or self._paused:
            return
        if len(data) < 16:
            return

        # RTP version check: version 2, payload type 0x78 (120) for voice
        if (data[0] >> 6) != 2:
            return
        if (data[1] & 0x7F) != 0x78:
            return

        first_byte = data[0]
        _, _, _, _, ssrc = struct.unpack_from(">BBHII", data, 0)

        if ssrc == self._bot_ssrc:
            return

        cc = first_byte & 0x0F
        has_extension = bool(first_byte & 0x10)
        has_padding = bool(first_byte & 0x20)
        header_size = 12 + (4 * cc) + (4 if has_extension else 0)

        if len(data) < header_size + 4:
            return

        header = bytes(data[:header_size])
        payload_with_nonce = data[header_size:]

        if len(payload_with_nonce) < 4:
            return

        nonce = bytearray(24)
        nonce[:4] = payload_with_nonce[-4:]
        encrypted = bytes(payload_with_nonce[:-4])

        # NaCl transport decrypt
        try:
            import nacl.secret
            box = nacl.secret.Aead(self._secret_key)
            decrypted = box.decrypt(encrypted, header, bytes(nonce))
        except Exception:
            return

        # Skip extension data
        if has_extension and hasattr(self, "_vc"):
            ext_size = 0
            ext_preamble_offset = 12 + (4 * cc)
            if len(data) > ext_preamble_offset + 4:
                ext_words = struct.unpack_from(">H", data, ext_preamble_offset + 2)[0]
                ext_size = ext_words * 4
            if ext_size and len(decrypted) > ext_size:
                decrypted = decrypted[ext_size:]

        # Strip RTP padding
        if has_padding:
            if not decrypted:
                return
            pad_len = decrypted[-1]
            if pad_len == 0 or pad_len > len(decrypted):
                return
            decrypted = decrypted[:-pad_len]
            if not decrypted:
                return

        # Opus decode
        try:
            if ssrc not in self._decoders:
                self._decoders[ssrc] = discord.opus.Decoder()
            pcm = self._decoders[ssrc].decode(decrypted)
            self._buffers[ssrc].extend(pcm)
            self._last_packet_time[ssrc] = time.monotonic()
        except Exception:
            pass

        # Map SSRC from SPEAKING events
        if ssrc not in self._ssrc_to_user:
            self._infer_user_for_ssrc(ssrc)

    def _infer_user_for_ssrc(self, ssrc: int) -> None:
        try:
            channel = self._vc.channel
            if not channel:
                return
            bot_id = self._vc.user.id if self._vc.user else 0
            candidates = [
                m.id for m in channel.members
                if m.id != bot_id and (not self._allowed_user_ids or str(m.id) in self._allowed_user_ids)
            ]
            if len(candidates) == 1:
                uid = candidates[0]
                self._ssrc_to_user[ssrc] = uid
        except Exception:
            pass

    def update_ssrc_mapping(self, user_id: int, ssrc: int) -> None:
        self._ssrc_to_user[ssrc] = user_id

    def check_silence(self) -> List[Tuple[int, bytes]]:
        """Return list of (user_id, pcm_bytes) for completed utterances."""
        now = time.monotonic()
        completed = []
        user_map = dict(self._ssrc_to_user)
        ssrc_list = list(self._buffers.keys())

        for ssrc in ssrc_list:
            last_time = self._last_packet_time.get(ssrc, now)
            silence_duration = now - last_time
            buf = self._buffers[ssrc]
            buf_duration = len(buf) / (_SAMPLE_RATE * _CHANNELS * 2)

            if silence_duration >= _SILENCE_THRESHOLD and buf_duration >= _MIN_SPEECH_DURATION:
                user_id = user_map.get(ssrc, 0)
                if user_id:
                    completed.append((user_id, bytes(buf)))
                self._buffers[ssrc] = bytearray()
                self._last_packet_time.pop(ssrc, None)
            elif silence_duration >= _SILENCE_THRESHOLD * 2:
                self._buffers.pop(ssrc, None)
                self._last_packet_time.pop(ssrc, None)

        return completed

    @staticmethod
    def pcm_to_wav(
        pcm_data: bytes,
        output_path: str,
        src_rate: int = 48000,
        src_channels: int = 2,
    ) -> None:
        """Convert raw 16-bit PCM to 16kHz mono WAV via ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as f:
            f.write(pcm_data)
            pcm_path = f.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "s16le",
                    "-ar", str(src_rate),
                    "-ac", str(src_channels),
                    "-i", pcm_path,
                    "-ar", "16000",
                    "-ac", "1",
                    output_path,
                ],
                check=True,
                timeout=10,
            )
        finally:
            try:
                os.unlink(pcm_path)
            except OSError:
                pass


class VoiceManager:
    """Manages Discord voice channel sessions per guild.

    Handles the lifecycle: join, listen for speech, transcribe, TTS response.
    """

    def __init__(
        self,
        app: "App",
        bot: discord.Client,
        cfg: Dict,
        allowed_user_ids: Set[str],
        get_agent_cb: Callable,
    ):
        self._app = app
        self._bot = bot
        self._cfg = cfg
        self._allowed_user_ids = allowed_user_ids
        self._get_agent = get_agent_cb

        self._voice_clients: Dict[int, discord.VoiceClient] = {}
        self._receivers: Dict[int, _VoiceReceiver] = {}
        self._listen_tasks: Dict[int, asyncio.Task] = {}
        self._timeout_tasks: Dict[int, asyncio.Task] = {}
        self._voice_text_channels: Dict[int, int] = {}  # guild_id → text_channel_id
        self._locks: Dict[int, asyncio.Lock] = {}
        self._speaking_tasks: Dict[int, asyncio.Task] = {}

        self._enabled = cfg.get("enabled", True)
        self._silence_threshold = float(cfg.get("silence_threshold", _SILENCE_THRESHOLD))
        self._min_speech_duration = float(cfg.get("min_speech_duration", _MIN_SPEECH_DURATION))
        self._voice_timeout = int(cfg.get("timeout", _VOICE_TIMEOUT))

    async def join(self, channel: discord.VoiceChannel, text_channel: discord.abc.Messageable) -> bool:
        """Join a Discord voice channel. Returns True on success."""
        if not self._enabled:
            logger.warning("Discord voice: voice support is disabled in config")
            return False

        guild_id = channel.guild.id
        lock = self._locks.setdefault(guild_id, asyncio.Lock())

        async with lock:
            existing = self._voice_clients.get(guild_id)
            if existing and existing.is_connected():
                if existing.channel and existing.channel.id == channel.id:
                    self._reset_timeout(guild_id)
                    return True
                try:
                    await existing.move_to(channel)
                    self._reset_timeout(guild_id)
                    return True
                except Exception as e:
                    logger.warning("Discord voice: move_to failed: %s", e)

            try:
                vc = await channel.connect()
            except Exception as e:
                logger.error("Discord voice: failed to connect to voice channel: %s", e)
                return False

            self._voice_clients[guild_id] = vc
            self._voice_text_channels[guild_id] = text_channel.id if hasattr(text_channel, "id") else 0
            self._reset_timeout(guild_id)

            # Start voice receiver
            try:
                receiver = _VoiceReceiver(vc, allowed_user_ids=self._allowed_user_ids)
                receiver.start()
                self._receivers[guild_id] = receiver
                self._listen_tasks[guild_id] = asyncio.ensure_future(self._listen_loop(guild_id, text_channel))
                logger.info("Discord voice: joined and listening in guild %d", guild_id)
            except Exception as e:
                logger.warning("Discord voice: receiver failed to start: %s", e)

            return True

    async def leave(self, guild_id: int) -> None:
        """Disconnect from the voice channel in a guild."""
        lock = self._locks.setdefault(guild_id, asyncio.Lock())

        async with lock:
            receiver = self._receivers.pop(guild_id, None)
            if receiver:
                receiver.stop()

            listen_task = self._listen_tasks.pop(guild_id, None)
            if listen_task:
                listen_task.cancel()

            vc = self._voice_clients.pop(guild_id, None)
            if vc and vc.is_connected():
                try:
                    await vc.disconnect()
                except Exception:
                    pass

            timeout_task = self._timeout_tasks.pop(guild_id, None)
            if timeout_task:
                timeout_task.cancel()

            self._voice_text_channels.pop(guild_id, None)

    async def cleanup_all(self) -> None:
        """Disconnect from all voice channels."""
        for guild_id in list(self._voice_clients.keys()):
            await self.leave(guild_id)

    def is_in_voice(self, guild_id: int) -> bool:
        vc = self._voice_clients.get(guild_id)
        return vc is not None and vc.is_connected()

    async def play_audio(self, guild_id: int, audio_path: str) -> bool:
        """Play an audio file in the connected voice channel."""
        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return False

        receiver = self._receivers.get(guild_id)
        if receiver:
            receiver.pause()

        try:
            # Wait for current playback to finish
            wait_start = time.monotonic()
            while vc.is_playing():
                if time.monotonic() - wait_start > _PLAYBACK_TIMEOUT:
                    logger.warning("Discord voice: timed out waiting for previous playback")
                    vc.stop()
                    break
                await asyncio.sleep(0.1)

            done = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _after(error):
                if error:
                    logger.error("Discord voice: playback error: %s", error)
                loop.call_soon_threadsafe(done.set)

            source = discord.FFmpegPCMAudio(audio_path)
            vc.play(source, after=_after)

            try:
                await asyncio.wait_for(done.wait(), timeout=_PLAYBACK_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("Discord voice: playback timed out after %ds", _PLAYBACK_TIMEOUT)
                vc.stop()

            self._reset_timeout(guild_id)
            return True
        finally:
            if receiver:
                receiver.resume()

    async def send_tts(self, guild_id: int, audio_bytes: bytes) -> bool:
        """Synthesize audio_bytes to a temp file and play it in the voice channel."""
        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return False

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            return await self.play_audio(guild_id, temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Handle voice state updates for SSRC mapping and auto-join."""
        guild_id = member.guild.id
        if guild_id not in self._voice_clients:
            return

        if after.channel and after.channel.id == self._voice_clients[guild_id].channel.id:
            # User joined our channel — handle SSRC mapping via SPEAKING
            if hasattr(after, "self_stream") or hasattr(after, "self_video"):
                pass  # SSRC mapping is handled in the speaking events
        elif before.channel and before.channel.id == self._voice_clients[guild_id].channel.id:
            # User left our channel
            pass

    def _reset_timeout(self, guild_id: int) -> None:
        task = self._timeout_tasks.pop(guild_id, None)
        if task:
            task.cancel()
        self._timeout_tasks[guild_id] = asyncio.ensure_future(self._timeout_handler(guild_id))

    async def _timeout_handler(self, guild_id: int) -> None:
        try:
            await asyncio.sleep(self._voice_timeout)
        except asyncio.CancelledError:
            return
        text_ch_id = self._voice_text_channels.get(guild_id)
        await self.leave(guild_id)
        if text_ch_id:
            channel = self._bot.get_channel(text_ch_id)
            if channel:
                try:
                    await channel.send("Voice-Channel verlassen (Inaktivität).")
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Voice listening loop
    # ------------------------------------------------------------------

    _KEEPALIVE_INTERVAL = 15  # seconds

    async def _listen_loop(self, guild_id: int, text_channel: discord.abc.Messageable) -> None:
        receiver = self._receivers.get(guild_id)
        if not receiver:
            return

        last_keepalive = time.monotonic()
        try:
            while receiver._running:
                await asyncio.sleep(0.2)

                # UDP keepalive
                now = time.monotonic()
                if now - last_keepalive >= self._KEEPALIVE_INTERVAL:
                    last_keepalive = now
                    try:
                        vc = self._voice_clients.get(guild_id)
                        if vc and vc.is_connected():
                            vc._connection.send_packet(b'\xf8\xff\xfe')
                    except Exception:
                        pass

                completed = receiver.check_silence()
                for user_id, pcm_data in completed:
                    asyncio.ensure_future(
                        self._process_voice(guild_id, user_id, pcm_data, text_channel)
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Discord voice: listen loop error: %s", e, exc_info=True)

    async def _process_voice(
        self,
        guild_id: int,
        user_id: int,
        pcm_data: bytes,
        text_channel: discord.abc.Messageable,
    ) -> None:
        # Convert PCM to WAV for STT
        tmp_f = tempfile.NamedTemporaryFile(suffix=".wav", prefix="discord_voice_", delete=False)
        wav_path = tmp_f.name
        tmp_f.close()

        try:
            await asyncio.to_thread(_VoiceReceiver.pcm_to_wav, pcm_data, wav_path)

            with open(wav_path, "rb") as f:
                wav_bytes = f.read()

            from pawlia.transcription import transcribe
            transcript = await transcribe(wav_bytes, self._app.config, mime="audio/wav")

            if not transcript:
                return

            logger.info("Discord voice: user %d said: %s", user_id, transcript[:120])

            # Show transcription in the text channel
            try:
                user = self._bot.get_user(user_id)
                username = user.display_name if user else f"User {user_id}"
                await text_channel.send(f":microphone2: **{username}**: *{transcript[:500]}*")
            except Exception:
                pass

            # Route to agent
            session_id = f"dc_{guild_id}_{text_channel.id}"
            agent = self._get_agent(guild_id, text_channel.id, None)

            app.scheduler.touch_activity(session_id)

            response = await agent.run(f"[Voice-Input von User {user_id}]: {transcript}")

            # Send text response
            if response:
                await text_channel.send(response)

            # TTS: synthesize and play
            from pawlia.tts import synthesize
            tts_audio = await synthesize(response[:500], self._app.config)
            if tts_audio:
                await self.send_tts(guild_id, tts_audio)

        except Exception as e:
            logger.error("Discord voice: processing error: %s", e, exc_info=True)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
