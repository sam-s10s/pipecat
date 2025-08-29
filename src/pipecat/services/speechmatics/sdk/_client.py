#
# Copyright (c) 2025, Speechmatics / Cantab Research Ltd
#

from __future__ import annotations

import asyncio
import datetime
import os
import re
from typing import Any, Optional

from loguru import logger
from speechmatics.rt import (
    AsyncClient,
    AudioFormat,
    ConversationConfig,
    ServerMessageType,
    TranscriptionConfig,
)

from ._models import (
    AgentServerMessageType,
    AnnotationFlags,
    DiarizationFocusMode,
    EndOfUtteranceMode,
    SpeakerFragmentView,
    SpeechFragment,
    VoiceAgentConfig,
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

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        config: Optional[VoiceAgentConfig] = None,
    ):
        """Initialize the Voice Agent client.

        Args:
            api_key: Speechmatics API key. If None, uses SPEECHMATICS_API_KEY env var.
            url: REST API endpoint URL. If None, uses SPEECHMATICS_BATCH_URL env var
                 or defaults to production endpoint.
            config: Voice agent configuration.
        """
        super().__init__(api_key=api_key, url=url)

        # Internal configuration
        self._transcription_config: Optional[TranscriptionConfig] = None
        self._audio_format: Optional[AudioFormat] = None

        # Connection status
        self._is_connected: bool = False

        # Timing info
        self._start_time: Optional[datetime.datetime] = None
        self._total_time: Optional[datetime.timedelta] = None
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
        self._end_of_utterance_mode: EndOfUtteranceMode = config.end_of_utterance_mode
        self._dz_config: Optional[DiarizationSpeakerConfig] = None

        # EndOfUtterance fallback timer
        self._finalize_task: Optional[asyncio.Task] = None

        # Segment processor
        self._processor_wait_time: float = 0.005
        self._processor_task: Optional[asyncio.Task] = None

        # Emitter task
        self._emitter_task: Optional[asyncio.Task] = None

        # Process the config
        self._process_config(config)
        self._register_event_handlers()

    def _process_config(self, config: Optional[VoiceAgentConfig] = None) -> None:
        """Create a formatted STT transcription and audio config.

        Creates a transcription config object based on the service parameters. Aligns
        with the Speechmatics RT API transcription config.
        """
        # Default
        if not config:
            config = VoiceAgentConfig()

        # Transcription config
        transcription_config = TranscriptionConfig(
            language=config.language,
            domain=config.domain,
            output_locale=config.output_locale,
            operating_point=config.operating_point,
            diarization="speaker" if config.enable_diarization else None,
            enable_partials=True,
            max_delay=config.max_delay,
        )

        # Additional vocab
        if config.additional_vocab:
            transcription_config.additional_vocab = [
                {
                    "content": e.content,
                    "sounds_like": e.sounds_like,
                }
                for e in config.additional_vocab
            ]

        # Diarization
        if config.enable_diarization:
            dz_cfg = {}
            if config.speaker_sensitivity is not None:
                dz_cfg["speaker_sensitivity"] = config.speaker_sensitivity
            if config.prefer_current_speaker is not None:
                dz_cfg["prefer_current_speaker"] = config.prefer_current_speaker
            if config.known_speakers:
                dz_cfg["speakers"] = {s.label: s.speaker_identifiers for s in config.known_speakers}
            if config.max_speakers is not None:
                dz_cfg["max_speakers"] = config.max_speakers
            if dz_cfg:
                transcription_config.speaker_diarization_config = dz_cfg

        # End of Utterance (for fixed)
        if (
            config.end_of_utterance_silence_trigger
            and config.end_of_utterance_mode == EndOfUtteranceMode.FIXED
        ):
            transcription_config.conversation_config = ConversationConfig(
                end_of_utterance_silence_trigger=config.end_of_utterance_silence_trigger,
            )

        # Punctuation overrides
        if config.punctuation_overrides:
            transcription_config.punctuation_overrides = config.punctuation_overrides

        # Set config
        self._transcription_config = transcription_config

        # Configure the audio
        self._audio_format = AudioFormat(
            encoding=config.audio_encoding,
            sample_rate=config.sample_rate,
            chunk_size=config.chunk_size,
        )

        # Diarization config
        self._dz_config = config.speaker_config

    def _register_event_handlers(self) -> None:
        """Register event handlers."""

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

        # End of utterance event
        if self._end_of_utterance_mode == EndOfUtteranceMode.FIXED:

            @self.on(ServerMessageType.END_OF_UTTERANCE)
            def _evt_on_end_of_utterance(message: dict[str, Any]):
                logger.warning("End of utterance detected")

        # End of Transcript
        @self.on(ServerMessageType.END_OF_TRANSCRIPT)
        def _evt_on_end_of_transcript(message: dict[str, Any]):
            pass

        # TODO - other events needed!

    async def connect(self) -> None:
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
                transcription_config=self._transcription_config, audio_format=self._audio_format
            )
            self._is_connected = True
        except Exception as e:
            logger.error(f"Exception: {e}")
            self.emit(
                AgentServerMessageType.ERROR,
                {"reason": "Unable to connect to API", "error": str(e)},
            )
            raise

    def update_diarization_config(self, config: DiarizationSpeakerConfig) -> None:
        """Update the diarization configuration."""
        self._dz_config = config

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
                            self._dz_config.focus_mode == DiarizationFocusMode.IGNORE
                            and self._dz_config.focus_speakers
                            and fragment.speaker not in self._dz_config.focus_speakers
                        ):
                            continue

                        # Drop ignored speakers
                        if (
                            self._dz_config.ignore_speakers
                            and fragment.speaker in self._dz_config.ignore_speakers
                        ):
                            continue

                    # Add the fragment
                    fragments.append(fragment)

            # Evaluate for VAD (only done on partials)
            if not is_final:
                self._vad_evaluation(fragments)

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
                focus_speakers=self._dz_config.focus_speakers,
                annotate_segments=False,
            )

            # New transcript
            tx_new = SpeakerFragmentView(
                fragments=self._speech_fragments.copy(),
                base_time=self._start_time,
                focus_speakers=self._dz_config.focus_speakers,
            )

            # Compare this against the previous transcript
            result = tx_new.annotate(tx_last)

            # Delay before emitting
            emit_final_delay: Optional[float] = None

            # If this is new, then copy over the new data
            if result.any(AnnotationFlags.NEW, AnnotationFlags.UPDATED_STRIPPED_LCASE):
                emit_final_delay = 0.5

            # If delay
            if emit_final_delay is not None:
                await self._emit_segments(tx_new, finalize_delay=emit_final_delay)

            # Copy the data
            self._last_speech_fragments = self._speech_fragments.copy()

    async def _emit_segments(self, view: SpeakerFragmentView, finalize_delay: float = 0.5) -> None:
        """Emit segments to listeners.

        Send speech segments to the pipeline. Will also emit speaking start / end events for VAD
        to be controlled by the client / framework.

        Args:
            view: The speaker fragment view to emit segments from.
            finalize_delay: Delay before finalizing partial segments.
        """
        # Clear the emitter timer
        if self._emitter_task is not None:
            self._emitter_task.cancel()

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

    def _vad_evaluation(self, fragments: list[SpeechFragment]):
        """Emit a VAD event.

        This will emit `SPEECH_STARTED` and `SPEECH_ENDED` events to the client and is
        based on valid transcription for active speakers. Ignored or speakers not in
        focus will not be considered an active participant.

        This should only run on partial / non-final words.

        Args:
            fragments: The list of fragments to use for evaluation.
        """
        # Check if any fragments are for active speakers
        partial_words = [frag for frag in fragments if not frag.is_final and frag._type == "word"]
        has_valid_partial = len(partial_words) > 0

        # No change required
        if has_valid_partial == self._is_speaking:
            return

        # Set the speaking state
        self._is_speaking = not self._is_speaking

        # Emit the event
        self.emit(
            AgentServerMessageType.SPEECH_STARTED
            if self._is_speaking
            else AgentServerMessageType.SPEECH_ENDED,
            {"words": len(partial_words)},
        )
