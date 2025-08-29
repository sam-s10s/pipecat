#
# Copyright (c) 2025, Speechmatics / Cantab Research Ltd
#

from __future__ import annotations

import asyncio
import datetime
import os
import re
from typing import Any

from loguru import logger
from speechmatics.rt import AudioFormat, ServerMessageType, TranscriptionConfig

from . import AsyncClient
from ._models import (
    AgentServerMessageType,
    AnnotationFlags,
    DiarizationFocusMode,
    SpeakerFragmentView,
    SpeechFragment,
)

DEBUG_MORE = os.getenv("SPEECHMATICS_DEBUG_MORE", "0").lower() in ["1", "true"]
DEBUG_MESSAGES = os.getenv("SPEECHMATICS_DEBUG_MESSAGES", "0").lower() in ["1", "true"]


class VoiceAgentClient(AsyncClient):
    """Voice Agent client.

    This class extends the AsyncClient class from the Speechmatics Real-Time SDK
    and provides additional functionality for processing partial and final
    transcription from the STT engine into accumulated transcriptions with
    flags to indicate changes between messages, etc.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Voice Agent client.

        Args:
            *args: All arguments passed to AsyncClient parent class.
            **kwargs: All arguments passed to AsyncClient parent class.
        """
        super().__init__(*args, **kwargs)

        # Connection status
        self._is_connected: bool = False

        # Timing info
        self._start_time: datetime.datetime | None = None
        self._total_time: datetime.timedelta | None = None
        self._total_bytes: int = 0

        # Time to disregard speech fragments before
        self._trim_before_time: float = 0.0

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []
        self._speech_fragments_lock: asyncio.Lock = asyncio.Lock()
        self._last_speech_fragments: list[SpeechFragment] = []

        # Speaking states
        self._is_speaking: bool = False

        # Speaker focus
        self._focus_speakers: list[str] | None = None
        self._ignore_speakers: list[str] | None = None
        self._focus_mode: DiarizationFocusMode = DiarizationFocusMode.RETAIN

        # EndOfUtterance fallback timer
        self._finalize_task: asyncio.Task | None = None

        # Segment processor
        self._processor_wait_time: float = 0.005
        self._processor_task: asyncio.Task | None = None

        # Emitter task
        self._emitter_task: asyncio.Task | None = None

        # Recognition started event
        @self.once(ServerMessageType.RECOGNITION_STARTED)
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

    async def _emit_segments(self, view: SpeakerFragmentView, finalize_delay: float = 0.5) -> None:
        """Emit segments to listeners.

        Send speech segments to the pipeline. If VAD is enabled, then this will
        also send an interruption and user started speaking frames. When the
        final transcript is received, then this will send a user stopped speaking
        and stop interruption frames.

        Args:
            view: The speaker fragment view to emit segments from.
            finalize_delay: Delay before finalizing partial segments.
        """
        # Clear the emitter timer
        if self._emitter_task is not None:
            self._emitter_task.cancel()

        print(view)

        # Emit interim results
        self.emit(AgentServerMessageType.INTERIM_SEGMENTS, {"segments": view.segments})

        # Emit finalized segments
        async def emit_after_delay(delay: float):
            # Wait for the delay (can be cancelled)
            if delay > 0:
                await asyncio.sleep(delay)

            # Emit the segments
            self.emit(AgentServerMessageType.FINAL_SEGMENTS, {"segments": view.segments})

            # Last end time
            self._trim_before_time = view.end_time + 0.01

            # Remove fragments that have been emitted
            self._speech_fragments = [
                fragment
                for fragment in self._speech_fragments
                if fragment.end_time > self._trim_before_time
            ]

            # Reset previous fragments
            self._last_speech_fragments = self._speech_fragments.copy()

            # Reset the task
            self._emitter_task = None

        # Start the timer
        self._emitter_task = asyncio.create_task(emit_after_delay(finalize_delay))

    def _handle_transcript(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle the partial and final transcript events.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.
        """
        # Handle async
        asyncio.create_task(self._handle_transcript_async(message, is_final))

    async def _handle_transcript_async(self, message: dict[str, Any], is_final: bool) -> None:
        """Handle the partial and final transcript events (async).

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.
        """
        # Debug payloads
        if DEBUG_MESSAGES:
            logger.debug(f"{message['message']}(message={message}, is_final={is_final})")

        # Add the speech fragments
        fragments_available = await self._add_speech_fragments(
            message=message,
            is_final=is_final,
        )

        # Skip if unchanged
        if not fragments_available:
            return

        # Clear any existing timer
        if self._processor_task is not None:
            self._processor_task.cancel()

        # Send transcription frames after delay
        async def process_after_delay(delay: float):
            await asyncio.sleep(delay)
            await self._process_speech_fragments()
            self._processor_task = None

        # Send frames after delay
        self._processor_task = asyncio.create_task(process_after_delay(self._processor_wait_time))

    async def _add_speech_fragments(self, message: dict[str, Any], is_final: bool = False) -> bool:
        """Takes a new Partial or Final from the STT engine.

        Accumulates it into the _speech_data list. As new final data is added, all
        partials are removed from the list.

        Note: If a known speaker is `__[A-Z0-9_]{2,}__`, then the words are skipped,
        as this is used to protect against self-interruption by the assistant or to
        block out specific known voices.

        Args:
            message: The new Partial or Final from the STT engine.
            is_final: Whether the data is final or partial.

        Returns:
            True if the speech fragments were updated, False otherwise.
        """
        async with self._speech_fragments_lock:
            # Parsed new speech data from the STT engine
            fragments: list[SpeechFragment] = []

            # Iterate over the results in the payload
            for result in message.get("results", []):
                alt = result.get("alternatives", [{}])[0]
                if alt.get("content", None):
                    # Create the new fragment
                    fragment = SpeechFragment(
                        start_time=result.get("start_time", 0),
                        end_time=result.get("end_time", 0),
                        language=alt.get("language", "en"),
                        _type=result.get("type", "word"),
                        is_eos=result.get("is_eos", False),
                        is_disfluency="disfluency" in alt.get("tags", []),
                        is_punctuation=result.get("type", "") == "punctuation",
                        is_final=is_final,
                        attaches_to=result.get("attaches_to", ""),
                        content=alt.get("content", ""),
                        speaker=alt.get("speaker", None),
                        confidence=alt.get("confidence", 1.0),
                        result=result,
                    )

                    # Check fragment is after trim time
                    if fragment.start_time < self._trim_before_time:
                        continue

                    # Speaker filtering
                    if fragment.speaker:
                        # Drop `__XX__` speakers
                        if re.match(r"^__[A-Z0-9_]{2,}__$", fragment.speaker):
                            continue

                        # Drop speakers not focussed on
                        if (
                            self._focus_mode == DiarizationFocusMode.IGNORE
                            and self._focus_speakers
                            and fragment.speaker not in self._focus_speakers
                        ):
                            continue

                        # Drop ignored speakers
                        if self._ignore_speakers and fragment.speaker in self._ignore_speakers:
                            continue

                    # Add the fragment
                    fragments.append(fragment)

            # Remove existing partials, as new partials and finals are provided
            self._speech_fragments = [frag for frag in self._speech_fragments if frag.is_final]

            # Add the fragments to the speech data
            self._speech_fragments.extend(fragments)

            # Fragments available
            return len(self._speech_fragments) > 0

    async def _process_speech_fragments(self) -> None:
        """Process the speech fragments.

        Compares the current speech fragments against the last set of speech fragments.
        When segments are emitted, they are then removed from the buffer of fragments
        so the next comparison is based on the remaining + new fragments.
        """
        async with self._speech_fragments_lock:
            # Current transcription
            tx_last = SpeakerFragmentView(
                fragments=self._last_speech_fragments.copy(),
                base_time=self._start_time,
                focus_speakers=self._focus_speakers,
                annotate_segments=False,
            )

            # New transcript
            tx_new = SpeakerFragmentView(
                fragments=self._speech_fragments.copy(),
                base_time=self._start_time,
                focus_speakers=self._focus_speakers,
            )

            # Compare this against the previous transcript
            result = tx_new.annotate(tx_last)

            # Delay before emitting
            emit_final_delay: float | None = None

            # If this is new, then copy over the new data
            if result.any(AnnotationFlags.NEW, AnnotationFlags.UPDATED_STRIPPED_LCASE):
                emit_final_delay = 0.5

            # If delay
            if emit_final_delay is not None:
                await self._emit_segments(tx_new, finalize_delay=emit_final_delay)

            # Copy the data
            self._last_speech_fragments = self._speech_fragments.copy()
