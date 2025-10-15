#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics STT service integration."""

import asyncio
import os
import time
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from pipecat import __version__ as pipecat_version
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from speechmatics.voice import (
        AdditionalVocabEntry,
        AgentClientMessageType,
        AgentServerMessageType,
        AudioEncoding,
        EndOfUtteranceMode,
        OperatingPoint,
        SpeakerFocusConfig,
        SpeakerFocusMode,
        SpeakerIdentifier,
        SpeechSegmentConfig,
        VoiceAgentClient,
        VoiceAgentConfig,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Speechmatics, you need to `pip install pipecat-ai[speechmatics]`."
    )
    raise Exception(f"Missing module: {e}")


__all__ = [
    "AdditionalVocabEntry",
    "EndOfUtteranceMode",
    "OperatingPoint",
    "SpeakerFocusConfig",
    "SpeakerFocusMode",
    "SpeakerIdentifier",
]


load_dotenv()


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.

    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, various audio formats,
    and speaker diarization.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Speechmatics STT service.

        Parameters:
            operating_point: Operating point for transcription accuracy vs. latency tradeoff. It is
                recommended to use OperatingPoint.ENHANCED for most use cases. Defaults to
                OperatingPoint.ENHANCED.

            domain: Domain for Speechmatics API. Defaults to None.

            language: Language code for transcription. Defaults to `Language.EN`.

            output_locale: Output locale for transcription, e.g. `Language.EN_GB`.
                Defaults to None.

            enable_vad: Enable VAD to trigger end of utterance detection. This should be used
                without any other VAD enabled in the agent and will emit the speaker started
                and stopped frames. Defaults to False.

            max_delay: Maximum delay in seconds for transcription. This forces the STT engine to
                speed up the processing of transcribed words and reduces the interval between partial
                and final results. Lower values can have an impact on accuracy. Defaults to 0.7.

            end_of_utterance_silence_trigger: Maximum delay in seconds for end of utterance trigger.
                The delay is used to wait for any further transcribed words before emitting the final
                word frames. The value must be lower than max_delay.
                Defaults to 0.2.

            end_of_utterance_max_delay: Maximum delay in seconds for end of utterance detection.
                This is used to wait for any further transcribed words before emitting the final
                word frames. The value must be lower than max_delay.
                Defaults to 10.0.

            end_of_utterance_mode: End of utterance delay mode. When ADAPTIVE is used, the delay
                can be adjusted on the content of what the most recent speaker has said, such as
                rate of speech and whether they have any pauses or disfluencies. When FIXED is used,
                the delay is fixed to the value of `end_of_utterance_delay`. Use of EXTERNAL disables
                end of utterance detection and uses end_of_utterance_max_delay as a fallback timer.
                Defaults to `EndOfUtteranceMode.FIXED`.

            additional_vocab: List of additional vocabulary entries. If you supply a list of
                additional vocabulary entries, the this will increase the weight of the words in the
                vocabulary and help the STT engine to better transcribe the words.
                Defaults to [].

            punctuation_overrides: Punctuation overrides. This allows you to override the punctuation
                in the STT engine. This is useful for languages that use different punctuation
                than English. See documentation for more information.
                Defaults to None.

            enable_diarization: Enable speaker diarization. When enabled, the STT engine will
                determine and attribute words to unique speakers. The speaker_sensitivity
                parameter can be used to adjust the sensitivity of diarization.
                Defaults to False.

            speaker_sensitivity: Diarization sensitivity. A higher value increases the sensitivity
                of diarization and helps when two or more speakers have similar voices.
                Defaults to 0.5.

            max_speakers: Maximum number of speakers to detect. This forces the STT engine to cluster
                words into a fixed number of speakers. It should not be used to limit the number of
                speakers, unless it is clear that there will only be a known number of speakers.
                Defaults to None.

            speaker_active_format: Formatter for active speaker ID. This formatter is used to format
                the text output for individual speakers and ensures that the context is clear for
                language models further down the pipeline. The attributes `text` and `speaker_id` are
                available. The system instructions for the language model may need to include any
                necessary instructions to handle the formatting.
                Example: `@{speaker_id}: {text}`.
                Defaults to transcription output.

            speaker_passive_format: Formatter for passive speaker ID. As with the
                speaker_active_format, the attributes `text` and `speaker_id` are available.
                Example: `@{speaker_id} [background]: {text}`.
                Defaults to transcription output.

            prefer_current_speaker: Prefer current speaker ID. When set to true, groups of words close
                together are given extra weight to be identified as the same speaker.
                Defaults to False.

            focus_speakers: List of speaker IDs to focus on. When enabled, only these speakers are
                emitted as finalized frames and other speakers are considered passive. Words from
                other speakers are still processed, but only emitted when a focussed speaker has
                also said new words. A list of labels (e.g. `S1`, `S2`) or identifiers of known
                speakers (e.g. `speaker_1`, `speaker_2`) can be used.
                Defaults to [].

            ignore_speakers: List of speaker IDs to ignore. When enabled, these speakers are
                excluded from the transcription and their words are not processed. Their speech
                will not trigger any VAD or end of utterance detection. By default, any speaker
                with a label starting and ending with double underscores will be excluded (e.g.
                `__ASSISTANT__`).
                Defaults to [].

            focus_mode: Speaker focus mode for diarization. When set to `SpeakerFocusMode.RETAIN`,
                the STT engine will retain words spoken by other speakers (not listed in `ignore_speakers`)
                and process them as passive speaker frames. When set to `SpeakerFocusMode.IGNORE`,
                the STT engine will ignore words spoken by other speakers and they will not be processed.
                Defaults to `SpeakerFocusMode.RETAIN`.

            known_speakers: List of known speaker labels and identifiers. If you supply a list of
                labels and identifiers for speakers, then the STT engine will use them to attribute
                any spoken words to that speaker. This is useful when you want to attribute words
                to a specific speaker, such as the assistant or a specific user. Labels and identifiers
                can be obtained from a running STT session and then used in subsequent sessions.
                Identifiers are unique to each Speechmatics account and cannot be used across accounts.
                Refer to our examples on the format of the known_speakers parameter.
                Defaults to [].

            enable_preview_features: Enable preview features using a preview endpoint provided by
                Speechmatics. Defaults to False.

            audio_encoding: Audio encoding format. Defaults to AudioEncoding.PCM_S16LE.
        """

        # Service configuration
        operating_point: OperatingPoint = OperatingPoint.ENHANCED
        domain: str | None = None
        language: Language | str = Language.EN
        output_locale: Language | str | None = None

        # Features
        enable_vad: bool = False
        max_delay: float = 0.7
        end_of_utterance_silence_trigger: float = 0.2
        end_of_utterance_max_delay: float = 10.0
        end_of_utterance_mode: EndOfUtteranceMode = EndOfUtteranceMode.FIXED
        additional_vocab: list[AdditionalVocabEntry] = []
        punctuation_overrides: dict | None = None

        # Diarization
        enable_diarization: bool = False
        speaker_sensitivity: float = 0.5
        max_speakers: int | None = None
        speaker_active_format: str = "{text}"
        speaker_passive_format: str | None = None
        prefer_current_speaker: bool = False
        focus_speakers: list[str] = []
        ignore_speakers: list[str] = []
        focus_mode: SpeakerFocusMode = SpeakerFocusMode.RETAIN
        known_speakers: list[SpeakerIdentifier] = []

        # Advanced features
        enable_preview_features: bool = False

        # Audio
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE

    class UpdateParams(BaseModel):
        """Update parameters for Speechmatics STT service.

        These are the only parameters that can be changed once a session has started. If you need to
        change the language, etc., then you must create a new instance of the service.

        Parameters:
            focus_speakers: List of speaker IDs to focus on. When enabled, only these speakers are
                emitted as finalized frames and other speakers are considered passive. Words from
                other speakers are still processed, but only emitted when a focussed speaker has
                also said new words. A list of labels (e.g. `S1`, `S2`) or identifiers of known
                speakers (e.g. `speaker_1`, `speaker_2`) can be used.
                Defaults to [].

            ignore_speakers: List of speaker IDs to ignore. When enabled, these speakers are
                excluded from the transcription and their words are not processed. Their speech
                will not trigger any VAD or end of utterance detection. By default, any speaker
                with a label starting and ending with double underscores will be excluded (e.g.
                `__ASSISTANT__`).
                Defaults to [].

            focus_mode: Speaker focus mode for diarization. When set to `SpeakerFocusMode.RETAIN`,
                the STT engine will retain words spoken by other speakers (not listed in `ignore_speakers`)
                and process them as passive speaker frames. When set to `SpeakerFocusMode.IGNORE`,
                the STT engine will ignore words spoken by other speakers and they will not be processed.
                Defaults to `SpeakerFocusMode.RETAIN`.
        """

        focus_speakers: list[str] = []
        ignore_speakers: list[str] = []
        focus_mode: SpeakerFocusMode = SpeakerFocusMode.RETAIN

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sample_rate: int = 16000,
        params: InputParams | None = None,
        **kwargs,
    ):
        """Initialize the Speechmatics STT service.

        Args:
            api_key: Speechmatics API key for authentication. Uses environment variable
                `SPEECHMATICS_API_KEY` if not provided.
            base_url: Base URL for Speechmatics API. Uses environment variable `SPEECHMATICS_RT_URL`
                or defaults to `wss://eu2.rt.speechmatics.com/v2`.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            params: Optional[InputParams]: Input parameters for the service.
            **kwargs: Additional arguments passed to STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Service parameters
        self._api_key: str = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self._base_url: str = (
            base_url or os.getenv("SPEECHMATICS_RT_URL") or "wss://eu2.rt.speechmatics.com/v2"
        )

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Deprecation check
        _check_deprecated_args(kwargs, params)

        # Voice agent
        self._client: VoiceAgentClient | None = None
        self._config: VoiceAgentConfig = self._prepare_config(sample_rate, params)

        # Outbound frame queue
        self._outbound_frames: asyncio.Queue[Frame] = asyncio.Queue()

        # Framework options
        self._enable_vad: bool = params.enable_vad
        self._speaker_active_format: str = params.speaker_active_format
        self._speaker_passive_format: str = (
            params.speaker_passive_format or params.speaker_active_format
        )

        # Metrics
        self.set_model_name(self._config.operating_point.value)

        # Message queue
        self._stt_msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stt_msg_task: asyncio.Task | None = None

        # Speaking states
        self._is_speaking: bool = False
        self._bot_speaking: bool = False

        # Event handlers
        if params.enable_diarization:
            self._register_event_handler("on_speakers_result")

    # ============================================================================
    # LIFE-CYCLE / SESSION MANAGEMENT
    # ============================================================================

    async def start(self, frame: StartFrame):
        """Called when the new session starts."""
        await super().start(frame)
        await self._connect()
        self._stt_msg_task = asyncio.create_task(self._process_stt_messages())

    async def stop(self, frame: EndFrame):
        """Called when the session ends."""
        if self._stt_msg_task and not self._stt_msg_task.done():
            self._stt_msg_task.cancel()
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Called when the session is cancelled."""
        if self._stt_msg_task and not self._stt_msg_task.done():
            self._stt_msg_task.cancel()
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self) -> None:
        """Connect to the STT service."""
        # Log the event
        logger.debug(f"{self} connecting to Speechmatics STT service")

        # STT client
        self._client: VoiceAgentClient = VoiceAgentClient(
            api_key=self._api_key,
            url=self._base_url,
            app=f"pipecat/{pipecat_version}",
            config=self._config,
        )

        # Add message queue
        def add_message(message: dict[str, Any]):
            self._stt_msg_queue.put_nowait(message)

        # Add listeners
        self._client.on(AgentServerMessageType.ADD_PARTIAL_SEGMENT, add_message)
        self._client.on(AgentServerMessageType.ADD_SEGMENT, add_message)
        self._client.on(AgentServerMessageType.TTFB_METRICS, add_message)

        # Add listeners for VAD
        if self._enable_vad:
            self._client.on(AgentServerMessageType.START_OF_TURN, add_message)
            self._client.on(AgentServerMessageType.END_OF_TURN, add_message)

        # Speaker result listener
        if self._config.enable_diarization:
            self._client.on(AgentServerMessageType.SPEAKERS_RESULT, add_message)

        # Connect to the client
        try:
            await self._client.connect()
            logger.debug(f"{self} connected to Speechmatics STT service")
        except Exception as e:
            logger.error(f"{self} error connecting to Speechmatics: {e}")
            self._client = None

    async def _disconnect(self) -> None:
        """Disconnect from the STT service."""
        # Disconnect the client
        logger.debug(f"{self} disconnecting from Speechmatics STT service")
        try:
            if self._client:
                await self._client.disconnect()
        except asyncio.TimeoutError:
            logger.warning(f"{self} timeout while closing Speechmatics client connection")
        except Exception as e:
            logger.error(f"{self} error closing Speechmatics client: {e}")
        finally:
            self._client = None
            await self._call_event_handler("on_disconnected")

    async def _process_stt_messages(self) -> None:
        """Process messages from the STT client."""
        try:
            while True:
                message = await self._stt_msg_queue.get()
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    def _prepare_config(self, sample_rate: int, params: InputParams | None = None) -> None:
        """Parse the InputParams into VoiceAgentConfig."""
        # Default config
        if not params:
            return VoiceAgentConfig()

        # Override defaults
        if params.end_of_utterance_mode == EndOfUtteranceMode.EXTERNAL:
            params.max_delay = 4.0
            params.end_of_utterance_max_delay = 10.0

        # Create config
        return VoiceAgentConfig(
            # Service
            operating_point=params.operating_point,
            domain=params.domain,
            language=_language_to_speechmatics_language(params.language),
            output_locale=_locale_to_speechmatics_locale(params.language, params.output_locale),
            # Features
            max_delay=params.max_delay,
            end_of_utterance_silence_trigger=params.end_of_utterance_silence_trigger,
            end_of_utterance_max_delay=params.end_of_utterance_max_delay,
            end_of_utterance_mode=params.end_of_utterance_mode,
            additional_vocab=params.additional_vocab,
            punctuation_overrides=params.punctuation_overrides,
            # Diarization
            enable_diarization=params.enable_diarization,
            speaker_sensitivity=params.speaker_sensitivity,
            max_speakers=params.max_speakers,
            prefer_current_speaker=params.prefer_current_speaker,
            speaker_config=SpeakerFocusConfig(
                focus_speakers=params.focus_speakers,
                ignore_speakers=params.ignore_speakers,
                focus_mode=params.focus_mode,
            ),
            known_speakers=params.known_speakers,
            # Advanced
            include_results=True,
            enable_preview_features=params.enable_preview_features,
            speech_segment_config=SpeechSegmentConfig(split_on_eos=False),
            # Audio
            sample_rate=sample_rate,
            audio_encoding=params.audio_encoding,
        )

    def update_params(
        self,
        params: UpdateParams,
    ) -> None:
        """Updates the speaker configuration.

        This can update the speakers to listen to or ignore during an in-flight
        transcription. Only available if diarization is enabled.

        Args:
            params: Update parameters for the service.
        """
        # Check possible
        if not self._config.enable_diarization:
            raise ValueError("Diarization is not enabled")

        # Update the existing diarization configuration
        if params.focus_speakers is not None:
            self._config.speaker_config.focus_speakers = params.focus_speakers
        if params.ignore_speakers is not None:
            self._config.speaker_config.ignore_speakers = params.ignore_speakers
        if params.focus_mode is not None:
            self._config.speaker_config.focus_mode = params.focus_mode

        # Send the update
        if self._client:
            self._client.update_diarization_config(self._config.speaker_config)

    # ============================================================================
    # HANDLE ENGINE MESSAGES
    # ============================================================================

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle a message from the STT client."""
        event = message.get("message", "")

        # Handle events
        match event:
            case AgentServerMessageType.ADD_PARTIAL_SEGMENT:
                await self._handle_partial_segment(message)
            case AgentServerMessageType.ADD_SEGMENT:
                await self._handle_segment(message)
            case AgentServerMessageType.START_OF_TURN:
                await self._handle_start_of_turn(message)
            case AgentServerMessageType.END_OF_TURN:
                await self._handle_end_of_turn(message)
            case AgentServerMessageType.SPEAKERS_RESULT:
                await self._handle_speakers_result(message)
            case AgentServerMessageType.TTFB_METRICS:
                await self._handle_ttfb_metrics(message)
            case _:
                logger.debug(f"Unhandled message: {event}")

    async def _handle_partial_segment(self, message: dict[str, Any]) -> None:
        """Handle AddPartialSegment events.

        AddPartialSegment events are triggered by Speechmatics STT when it detects a
        partial segment of speech. These events provide the partial transcript for
        the current speaking turn.

        Args:
            message: the message payload.
        """
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            await self._send_frames(segments)

    async def _handle_segment(self, message: dict[str, Any]) -> None:
        """Handle AddSegment events.

        AddSegment events are triggered by Speechmatics STT when it detects a
        final segment of speech. These events provide the final transcript for
        the current speaking turn.

        Args:
            message: the message payload.
        """
        segments: list[dict[str, Any]] = message.get("segments", [])
        if segments:
            await self._send_frames(segments, finalized=True)

    async def _handle_start_of_turn(self, message: dict[str, Any]) -> None:
        """Handle StartOfTurn events.

        When Speechmatics STT detects the start of a new speaking turn, a StartOfTurn
        event is triggered. This triggers bot interruption to stop any ongoing speech
        synthesis and signals the start of user speech detection.

        The service will:
        - Send a BotInterruptionFrame upstream to stop bot speech
        - Send a UserStartedSpeakingFrame downstream to notify other components
        - Start metrics collection for measuring response times

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} User {message.get('speaker_id') or 'UU'} started speaking")
        await self.push_interruption_task_frame_and_wait()
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await self.start_processing_metrics()

    async def _handle_end_of_turn(self, message: dict[str, Any]) -> None:
        """Handle EndOfTurn events.

        EndOfTurn events are triggered by Speechmatics STT when it concludes a
        speaking turn. This occurs either due to silence or reaching the
        end-of-turn confidence thresholds. These events provide the final
        transcript for the completed turn.

        The service will:
        - Stop processing metrics collection
        - Send a UserStoppedSpeakingFrame to signal turn completion

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} User {message.get('speaker_id') or 'UU'} stopped speaking")
        await self.stop_processing_metrics()
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _handle_speakers_result(self, message: dict[str, Any]) -> None:
        """Handle SpeakersResult events.

        SpeakersResult events are triggered by Speechmatics STT when it provides
        speaker information for the current speaking turn.

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} speakers result received from STT")
        await self._call_event_handler("on_speakers_result", message)

    # ============================================================================
    # SEND FRAMES TO PIPELINE
    # ============================================================================

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for VAD and metrics handling.

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        # Forward to parent
        await super().process_frame(frame, direction)

        # Track the bot
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        # Force finalization
        if isinstance(frame, UserStoppedSpeakingFrame):
            if not self._enable_vad and self._client is not None:
                self._client.finalize()

    async def _send_frames(self, segments: list[dict[str, Any]], finalized: bool = False) -> None:
        """Send frames to the pipeline.

        Args:
            segments: The segments to send.
            finalized: Whether the data is final or partial.
        """
        # Skip if no frames
        if not segments:
            return

        # Frames to send
        frames: list[Frame] = []

        # Create frame from segment
        def attr_from_segment(segment: dict[str, Any]) -> dict[str, Any]:
            # Formats the output text based on the speaker and defined formats from the config.
            text = (
                self._speaker_active_format
                if segment.get("is_active", True)
                else self._speaker_passive_format
            ).format(
                **{
                    "speaker_id": segment.get("speaker_id", "UU"),
                    "text": segment.get("text", ""),
                    "ts": segment.get("timestamp"),
                    "lang": segment.get("language"),
                }
            )

            # Return the attributes for the frame
            return {
                "text": text,
                "user_id": segment.get("speaker_id") or "",
                "timestamp": segment.get("timestamp"),
                "language": segment.get("language"),
                "result": segment.get("results", []),
            }

        # If final, then re-parse into TranscriptionFrame
        if finalized:
            frames += [TranscriptionFrame(**attr_from_segment(segment)) for segment in segments]
            finalized_text = "|".join([s["text"] for s in segments])
            await self._handle_transcription(finalized_text, True, segments[0]["language"])
            logger.debug(f"{self} finalized transcript: {[f.text for f in frames]}")

        # Return as interim results (unformatted)
        else:
            frames += [
                InterimTranscriptionFrame(**attr_from_segment(segment)) for segment in segments
            ]
            logger.debug(f"{self} interim transcript: {[f.text for f in frames]}")

        # Send the frames
        for frame in frames:
            await self.push_frame(frame)

    # ============================================================================
    # PUBLIC FUNCTIONS
    # ============================================================================

    async def send_message(self, message: AgentClientMessageType | str, **kwargs: Any) -> None:
        """Send a message to the STT service.

        This sends a message to the STT service via the underlying transport. If the session
        is not running, this will raise an exception. Messages in the wrong format will also
        cause an error.

        Args:
            message: Message to send to the STT service.
            **kwargs: Additional arguments passed to the underlying transport.
        """
        try:
            payload = {"message": message}
            payload.update(kwargs)
            logger.debug(f"{self} sending message to STT: {payload}")
            asyncio.run_coroutine_threadsafe(
                self._client.send_message(payload), self.get_event_loop()
            )
        except Exception as e:
            raise RuntimeError(f"{self} error sending message to STT: {e}")

    # ============================================================================
    # METRICS
    # ============================================================================

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechmatics STT supports generation of metrics.
        """
        return True

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Adds audio to the audio buffer and yields None."""
        try:
            if self._client:
                await self._client.send_audio(audio)
            yield None
        except Exception as e:
            logger.error(f"Speechmatics error: {e}")
            yield ErrorFrame(f"Speechmatics error: {e}", fatal=False)
            await self._disconnect()

    async def _handle_ttfb_metrics(self, message: dict[str, Any]) -> None:
        """Handle TTFB metrics events.

        TTFB metrics events are triggered by Speechmatics STT when it provides
        metrics for the current speaking turn.

        Args:
            message: the message payload.
        """
        await self._emit_ttfb_metrics(message.get("ttfb"))

    async def _emit_ttfb_metrics(self, ttfb: int) -> None:
        """Create TTFB metrics.

        The TTFB is the miliseconds between the person speaking and the STT
        engine emitting the first partial. This is only calculated at the
        start of an utterance.
        """
        # Skip if metrics not available
        if not self._metrics:
            return

        # Calculate time as time.time() - ttfb (which is seconds)
        start_time = time.time() - (ttfb / 1000.0)

        # TODO - this does not use official methods!

        # Update internal metrics
        self._metrics._start_ttfb_time = start_time
        self._metrics._start_processing_time = start_time

        # Stop TTFB metrics
        await self.stop_ttfb_metrics()

        # TODO - have this elsewhere to capture all processing time?
        await self.stop_processing_metrics()


# ============================================================================
# HELPERS
# ============================================================================


def _language_to_speechmatics_language(language: Language) -> str:
    """Convert a Language enum to a Speechmatics language code.

    Args:
        language: The Language enum to convert.

    Returns:
        str: The Speechmatics language code, if found.
    """
    # List of supported input languages
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BA: "ba",
        Language.EU: "eu",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.YUE: "yue",
        Language.CA: "ca",
        Language.HR: "hr",
        Language.CS: "cs",
        Language.DA: "da",
        Language.NL: "nl",
        Language.EN: "en",
        Language.EO: "eo",
        Language.ET: "et",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.DE: "de",
        Language.EL: "el",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.IT: "it",
        Language.ID: "id",
        Language.GA: "ga",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.LV: "lv",
        Language.LT: "lt",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.CMN: "cmn",
        Language.MR: "mr",
        Language.MN: "mn",
        Language.NO: "no",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.ES: "es",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UG: "ug",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.CY: "cy",
    }

    # Get the language code
    result = BASE_LANGUAGES.get(language)

    # Fail if language is not supported
    if not result:
        raise ValueError(f"Unsupported language: {language}")

    # Return the language code
    return result


def _locale_to_speechmatics_locale(language_code: str, locale: Language) -> str | None:
    """Convert a Language enum to a Speechmatics language code.

    Args:
        language_code: The language code.
        locale: The Language enum to convert.

    Returns:
        str: The Speechmatics language code, if found.
    """
    # Languages and output locales
    LOCALES = {
        "en": {
            Language.EN_GB: "en-GB",
            Language.EN_US: "en-US",
            Language.EN_AU: "en-AU",
        },
    }

    # Get the locale code
    result = LOCALES.get(language_code, {}).get(locale)

    # Fail if locale is not supported
    if not result:
        logger.warning(f"Unsupported output locale: {locale}, defaulting to {language_code}")

    # Return the locale code
    return result


def _check_deprecated_args(kwargs: dict, params: SpeechmaticsSTTService.InputParams) -> None:
    """Check arguments for deprecation and update params if necessary.

    This function will show deprecation warnings for deprecated arguments and
    migrate them to the new location in the params object. If the new location
    is None, the argument is not used.

    Args:
        kwargs: Keyword arguments passed to the constructor.
        params: Input parameters for the service.
    """

    # Show deprecation warnings
    def _deprecation_warning(old: str, new: str | None = None):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            if new:
                message = f"`{old}` is deprecated, use `InputParams.{new}`"
            else:
                message = f"`{old}` is deprecated and not used"
            warnings.warn(message, DeprecationWarning)

    # List of deprecated arguments and their new location
    deprecated_args = [
        ("language", "language"),
        ("language_code", "language"),
        ("domain", "domain"),
        ("output_locale", "output_locale"),
        ("output_locale_code", "output_locale"),
        ("enable_partials", None),
        ("max_delay", "max_delay"),
        ("chunk_size", "chunk_size"),
        ("audio_encoding", "audio_encoding"),
        ("end_of_utterance_silence_trigger", "end_of_utterance_silence_trigger"),
        {"enable_speaker_diarization", "enable_diarization"},
        ("text_format", "speaker_active_format"),
        ("max_speakers", "max_speakers"),
        ("transcription_config", None),
    ]

    # Show warnings + migrate the arguments
    for old, new in deprecated_args:
        if old in kwargs:
            _deprecation_warning(old, new)
            if kwargs.get(old, None) is not None:
                params.__setattr__(new, kwargs[old])
