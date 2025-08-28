#
# Copyright (c) 2025, Speechmatics / Cantab Research Ltd
#

import asyncio
import datetime
from typing import Any

from loguru import logger
from speechmatics.rt import AudioFormat, ServerMessageType, TranscriptionConfig

from . import AsyncClient
from ._models import AgentServerMessageType, SpeechFragment


class VoiceAgentClient(AsyncClient):
    """Voice Agent client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Connection status
        self._is_connected: bool = False

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []
        self._speech_fragments_lock: asyncio.Lock = asyncio.Lock()

        # Speaking states
        self._is_speaking: bool = False

        # Timing info
        self._start_time: datetime.datetime | None = None
        self._total_time: datetime.timedelta | None = None
        self._total_bytes: int = 0

        # EndOfUtterance fallback timer
        self._finalize_timer: asyncio.Task | None = None

        # Frame sender timer
        self._transcription_frame_sender_wait_time: float = 0.005
        self._transcription_frame_sender_timer: asyncio.Task | None = None

        # Recognition started event
        @self.once(AgentServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]):
            logger.debug(f"Recognition started (session: {message.get('id')})")
            self._start_time = datetime.datetime.now(datetime.timezone.utc)

        # Partial transcript event
        @self.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
        def _evt_on_partial_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=False)

        # Final transcript event
        @self.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=True)

    async def connect(
        self, transcription_config: TranscriptionConfig, audio_format: AudioFormat
    ) -> None:
        """Connect to the Speechmatics API.

        Args:
            transcription_config: Transcription configuration.
            audio_format: Audio format.
        """
        # Check if we are already connected
        if self._is_connected:
            self.emit(
                AgentServerMessageType.ERROR,
                {"reason": "Already connected"},
            )
            return

        # Log the event
        logger.debug(f"{self} Connecting to Speechmatics STT service")

        # Connect to API
        try:
            await self.start_session(
                transcription_config=transcription_config, audio_format=audio_format
            )
            self._is_connected = True
        except Exception as e:
            logger.error(f"Exception: {e}")
            self.emit(
                AgentServerMessageType.ERROR,
                {"reason": "Unable to connect to API", "error": str(e)},
            )

    def _handle_transcript(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle the partial and final transcript events.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.
        """
        # Simple debug
        logger.debug(f"{is_final} -> {message}")

        # Emit a test event
        self.emit(
            AgentServerMessageType.FINAL_SEGMENT
            if is_final
            else AgentServerMessageType.INTERIM_SEGMENT,
            message,
        )
