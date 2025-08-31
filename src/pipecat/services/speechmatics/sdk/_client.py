#
# Copyright (c) 2025, Speechmatics / Cantab Research Ltd
#

from __future__ import annotations

import asyncio
import datetime
import os
import re
import time
from typing import Any, Optional

from speechmatics.rt import (
    AsyncClient,
    AudioFormat,
    ConversationConfig,
    ServerMessageType,
    TranscriptionConfig,
)

from ._logging import get_logger
from ._models import (
    AgentServerMessageType,
    AnnotationFlags,
    DiarizationFocusMode,
    EndOfUtteranceMode,
    SpeakerFragmentView,
    SpeakerVADStatus,
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

        # Logger
        self._logger = get_logger(__name__)

        # Internal configuration
        self._transcription_config: Optional[TranscriptionConfig] = None
        self._audio_format: Optional[AudioFormat] = None

        # Connection status
        self._is_connected: bool = False
        self._is_ready_for_audio: bool = False

        # Timing info
        self._start_time: Optional[datetime.datetime] = None
        self._total_time: float = 0
        self._total_bytes: int = 0

        # TTFB metrics
        self._last_ttfb_time: Optional[float] = None
        self._last_ttfb: int = 0

        # Time to disregard speech fragments before
        self._trim_before_time: float = 0

        # Current utterance speech data
        self._speech_fragments: list[SpeechFragment] = []
        self._speech_fragments_lock: asyncio.Lock = asyncio.Lock()
        self._last_speech_fragments: list[SpeechFragment] = []

        # Speaking states
        self._is_speaking: bool = False
        self._current_speaker: Optional[str] = None
        self._last_vad_time: float = 0

        # Speaker focus
        self._end_of_utterance_mode: EndOfUtteranceMode = config.end_of_utterance_mode
        self._end_of_utterance_delay: float = config.end_of_utterance_silence_trigger
        self._dz_enabled: bool = config.enable_diarization
        self._dz_config: Optional[DiarizationSpeakerConfig] = None

        # EndOfUtterance fallback timer
        self._finalize_task: Optional[asyncio.Task] = None

        # Segment processor
        self._processor_wait_time: float = 0.005
        self._processor_task: Optional[asyncio.Task] = None

        # Emitter task
        self._emitter_task: Optional[asyncio.Task] = None

        # Metrics emitter task
        self._metrics_emitter_interval: float = 10.0
        self._metrics_emitter_task: Optional[asyncio.Task] = None

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
        """Register event handlers.

        Specific event handlers that we need to deal with. All other events
        from the STT API will be available to clients to use themselves.
        """

        # Recognition started event
        @self.once(ServerMessageType.RECOGNITION_STARTED)
        def _evt_on_recognition_started(message: dict[str, Any]):
            self._logger.debug(f"Recognition started (session: {message.get('id')})")
            self._start_time = datetime.datetime.now(datetime.timezone.utc)
            self._is_ready_for_audio = True

        # Partial transcript event
        @self.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
        def _evt_on_partial_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=False)

        # Final transcript event
        @self.on(ServerMessageType.ADD_TRANSCRIPT)
        def _evt_on_final_transcript(message: dict[str, Any]):
            self._handle_transcript(message, is_final=True)

        # TODO - this should not be needed, as it forces finals internally?
        # # End of utterance event
        # if self._end_of_utterance_mode == EndOfUtteranceMode.FIXED:
        #     @self.on(ServerMessageType.END_OF_UTTERANCE)
        #     def _evt_on_end_of_utterance(message: dict[str, Any]):
        #         self._logger.warning("End of utterance detected - **TODO**")

        # End of Transcript
        @self.on(ServerMessageType.END_OF_TRANSCRIPT)
        def _evt_on_end_of_transcript(message: dict[str, Any]):
            pass

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

        # Connect to API
        try:
            await self.start_session(
                transcription_config=self._transcription_config, audio_format=self._audio_format
            )
            self._is_connected = True
            self._start_metrics_task()
        except Exception as e:
            self._logger.error(f"Exception: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the Speechmatics API."""
        # Check if we are already connected
        if not self._is_connected:
            return

        # Disconnect from API
        try:
            await asyncio.wait_for(self.close(), timeout=5.0)
        except asyncio.TimeoutError:
            self._logger.warning(f"{self} Timeout while closing Speechmatics client connection")
            raise
        except Exception as e:
            self._logger.error(f"{self} Error closing Speechmatics client: {e}")
            raise
        finally:
            self._is_connected = False
            self._is_ready_for_audio = False
            self._stop_metrics_task()

    def update_diarization_config(self, config: DiarizationSpeakerConfig) -> None:
        """Update the diarization configuration.

        You can update the speakers that needs to be focussed on or ignored during
        a session. The new config will overwrite the existing configuration and become
        active immediately.

        Args:
            config: The new diarization configuration.
        """
        self._dz_config = config

    async def send_audio(self, payload: bytes) -> None:
        """Send an audio frame through the WebSocket.

        Args:
            payload: The audio frame to send.

        Examples:
            >>> audio_chunk = b""
            >>> await client.send_audio(audio_chunk)
        """
        # Skip if not ready for audio
        if not self._is_ready_for_audio:
            return

        # Send to the AsyncClient
        await super().send_audio(payload)

        # Calculate the time (in seconds) for the payload
        self._total_time += len(payload) / self._audio_format.sample_rate / 2
        self._total_bytes += len(payload)

    def _start_metrics_task(self) -> None:
        """Start the metrics task."""

        # Task to send metrics
        async def emit_metrics() -> None:
            while True:
                # Interval between emitting metrics
                await asyncio.sleep(self._metrics_emitter_interval)

                # Check if there are any listeners for AgentServerMessageType.METRICS
                if not self.listeners(AgentServerMessageType.METRICS):
                    break

                # Calculations
                time_s = round(self._total_time, 3)

                # Emit metrics
                try:
                    self.emit(
                        AgentServerMessageType.METRICS,
                        {
                            "total_time": time_s,
                            "total_time_str": time.strftime("%H:%M:%S", time.gmtime(time_s)),
                            "total_bytes": self._total_bytes,
                            "last_ttfb": self._last_ttfb,
                        },
                    )
                except Exception:
                    pass

        # Trigger the task
        self._metrics_emitter_task = asyncio.create_task(emit_metrics())

    def _stop_metrics_task(self) -> None:
        """Stop the metrics task."""
        if self._metrics_emitter_task:
            self._metrics_emitter_task.cancel()
            self._metrics_emitter_task = None

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
            self._logger.debug(f"{message['message']}(message={message}, is_final={is_final})")

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

    def _calculate_ttfb(self, end_time: float) -> None:
        """Calculate the time to first text.

        The TTFB is calculated by taking the end time of the payload from the STT
        engine and then calculating the difference between the total time of bytes
        sent to the engine from the client.

        Args:
            end_time: The end time of the payload from the STT engine.
        """
        # Skip if no valid fragments
        if not self._speech_fragments:
            return

        # Get start of the first fragment
        start_time = self._speech_fragments[0].start_time

        # Skip if no partial word or if we have already calculated the TTFB for this word
        if start_time == self._last_ttfb_time:
            return

        # Calculate the time difference (convert to ms)
        self._last_ttfb = round((self._total_time - end_time) * 1000)
        self._last_ttfb_time = start_time

        # Debug
        self._logger.debug(f"TTFB {self._total_time} - {start_time} = {self._last_ttfb}")

        # Skip if no listeners
        if not self.listeners(AgentServerMessageType.TTFB_METRICS):
            return

        # Emit the TTFB
        async def emit_ttfb() -> None:
            try:
                self.emit(
                    AgentServerMessageType.TTFB_METRICS,
                    {
                        "ttfb": self._last_ttfb,
                    },
                )
            except Exception:
                pass

        # Trigger the task
        asyncio.create_task(emit_ttfb())

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

            # Update TTFB
            if not is_final:
                self._calculate_ttfb(end_time=message.get("metadata", {}).get("end_time", 0))

            # Fragments available
            return len(self._speech_fragments) > 0

    async def _process_speech_fragments(
        self, finalize: bool = False, emit_final_delay: Optional[float] = None
    ) -> None:
        """Process the speech fragments.

        Compares the current speech fragments against the last set of speech fragments.
        When segments are emitted, they are then removed from the buffer of fragments
        so the next comparison is based on the remaining + new fragments.

        Args:
            finalize: Optional override to emit segments, even if no changes.
            emit_final_delay: Optional delay before finalizing partial segments.
        """
        async with self._speech_fragments_lock:
            # Current transcription
            last_view = SpeakerFragmentView(
                fragments=self._last_speech_fragments.copy(),
                base_time=self._start_time,
                focus_speakers=self._dz_config.focus_speakers,
                annotate_segments=False,
            )

            # New transcript
            current_view = SpeakerFragmentView(
                fragments=self._speech_fragments.copy(),
                base_time=self._start_time,
                focus_speakers=self._dz_config.focus_speakers,
            )

            # Compare this against the previous transcript
            annotation_result = current_view.annotate(last_view)

            # Emit segments
            should_emit: bool = False

            # TODO - Consider a config option to split segments into smaller chunks for long transcripts?

            # Force segments to be emitted
            if finalize:
                should_emit = True
                if emit_final_delay is None:
                    emit_final_delay = 0.001

            # Calculate the delay
            else:
                should_emit, emit_final_delay = self._calculate_final_delay(
                    view=current_view, annotation=annotation_result
                )

            # Emit segments
            if should_emit:
                asyncio.create_task(
                    self._emit_segments(current_view, finalize_delay=emit_final_delay)
                )

            # Copy the data
            self._last_speech_fragments = self._speech_fragments.copy()

    def _calculate_final_delay(
        self,
        view: SpeakerFragmentView,
        annotation: AnnotationResult,
    ) -> Tuple(bool, float):
        """Calculate the delay before finalizing segments.

        Process the most recent segment and view to determine how long to delay before emitting
        the segments to the client.

        Args:
            view: The speaker fragment view to emit segments from.
            annotation: The annotation result to emit segments from.

        Returns:
            Tuple of (should_emit, emit_final_delay)
        """
        # Emit segments
        emit_final_delay: Optional[float] = None
        should_emit: bool = False

        # Last active segment
        last_active_segment_index = view.last_active_segment_index
        last_active_segment = (
            view.segments[last_active_segment_index] if last_active_segment_index > -1 else None
        )

        # If this is NEW or UPDATED_STRIPPED_LCASE
        if annotation.any(AnnotationFlags.NEW, AnnotationFlags.UPDATED_STRIPPED_LCASE):
            """Process the annotation flags to determine how long before sending a final segment."""

            # Last segment ends with EOS, emit final immediately
            if last_active_segment and last_active_segment.annotation.has(
                AnnotationFlags.ENDS_WITH_EOS
            ):
                emit_final_delay = 0.001

            # Fallback when using FIXED
            elif self._end_of_utterance_mode == EndOfUtteranceMode.FIXED:
                emit_final_delay = self._end_of_utterance_delay * 1.5

            # Timer for when ADAPTIVE
            elif self._end_of_utterance_mode == EndOfUtteranceMode.ADAPTIVE:
                emit_final_delay = self._end_of_utterance_delay

            # Emit segments
            should_emit = True

        # TODO - Other checks / end of turn detection

        # Return the result
        return should_emit, emit_final_delay

    def finalize_segments(self, emit_final_delay: Optional[float] = None):
        """Finalize segments.

        This function will emit segments in the buffer without any further checks
        on the contents of the segments. It should only be used if the end of utterance
        mode has been set to `NONE`.

        Args:
            emit_final_delay: Optional delay before finalizing partial segments.
        """
        asyncio.create_task(
            self._process_speech_fragments(finalize=True, emit_final_delay=emit_final_delay)
        )

    async def _emit_segments(
        self, view: SpeakerFragmentView, finalize_delay: Optional[float] = None
    ) -> None:
        """Emit segments to listeners.

        Send speech segments to the pipeline. Will also emit speaking start / end events for VAD
        to be controlled by the client / framework.

        Args:
            view: The speaker fragment view to emit segments from.
            finalize_delay: Optional delay before finalizing partial segments.
        """
        # Clear the emitter timer
        if self._emitter_task is not None:
            self._emitter_task.cancel()

        # Emit interim results
        try:
            self.emit(AgentServerMessageType.INTERIM_SEGMENTS, {"segments": view.segments})
        except Exception:
            pass

        # Emit finalized segments
        async def emit_after_delay(delay: float):
            # Wait for the delay (can be cancelled)
            if delay > 0:
                await asyncio.sleep(delay)

            # Find out if we have segments to emit
            segments_to_emit = view.last_active_segment_index + 1

            # Only finalize if there are valid segments
            if segments_to_emit > 0:
                # Number of segments held back
                segments_held_back = view.segment_count - segments_to_emit

                # Log if some segments are missing
                if segments_held_back:
                    self._logger.debug(
                        f"Holding segments: {segments_held_back} of {view.segment_count}"
                    )

                # Accumulate all of the fragments for segments up to and including the last_active_index
                fragments_to_emit = []
                for i in range(segments_to_emit):
                    fragments_to_emit.extend(view.segments[i].fragments)

                # Only include segments up to the last active segment
                trimmed_view = SpeakerFragmentView(
                    fragments=fragments_to_emit,
                    base_time=view.base_time,
                    focus_speakers=view.focus_speakers,
                    annotate_segments=False,
                )

                # Emit the segments
                try:
                    self.emit(
                        AgentServerMessageType.FINAL_SEGMENTS, {"segments": trimmed_view.segments}
                    )
                except Exception:
                    pass

                # Last end time
                self._trim_before_time = trimmed_view.end_time + 0.01

                # Filter segments after the trim time
                self._speech_fragments = [
                    fragment
                    for fragment in self._speech_fragments
                    if fragment.start_time >= self._trim_before_time
                ]

                # Reset previous fragments
                self._last_speech_fragments = self._speech_fragments.copy()

            # Reset the task
            self._emitter_task = None

        # Start the timer
        if finalize_delay is not None:
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
        # Find the valid list of partial words
        if self._dz_enabled and self._dz_config.focus_speakers:
            partial_words = [
                frag
                for frag in fragments
                if frag.speaker in self._dz_config.focus_speakers
                and frag._type == "word"
                and not frag.is_final
            ]
        else:
            partial_words = [
                frag for frag in fragments if frag._type == "word" and not frag.is_final
            ]

        # Evaluate if any valid partial words exist
        has_valid_partial = len(partial_words) > 0

        # Are we already speaking
        already_speaking = self._is_speaking

        # Speakers
        current_speaker = self._current_speaker
        speaker = partial_words[-1].speaker if has_valid_partial else self._current_speaker
        speaker_changed = speaker != current_speaker and current_speaker is not None

        # If diarization is enabled, indicate speaker switching
        if self._dz_enabled and speaker is not None:
            """When enabled, we send a speech events if the speaker has changed.
            
            For any client that wishes to show _which_ speaker is speaking, this will
            emit events to indicate when speakers switch.
            """

            # Check if speaker is different to the current speaker
            if already_speaking and speaker_changed:
                try:
                    self.emit(
                        AgentServerMessageType.SPEECH_ENDED,
                        SpeakerVADStatus(speaker_id=current_speaker, is_active=False),
                    )
                    self.emit(
                        AgentServerMessageType.SPEECH_STARTED,
                        SpeakerVADStatus(speaker_id=speaker, is_active=True),
                    )
                except Exception:
                    pass

        # Update current speaker
        self._current_speaker = speaker

        # No change required
        if has_valid_partial == already_speaking:
            return

        # Set the speaking state
        self._is_speaking = not self._is_speaking

        # Emit the event for latest speaker
        try:
            self.emit(
                AgentServerMessageType.SPEECH_STARTED
                if self._is_speaking
                else AgentServerMessageType.SPEECH_ENDED,
                SpeakerVADStatus(speaker_id=speaker, is_active=self._is_speaking),
            )
        except Exception:
            pass

        # Reset the current speaker
        if not self._is_speaking:
            self._current_speaker = None
