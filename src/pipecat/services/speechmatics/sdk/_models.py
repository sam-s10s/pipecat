#
# Copyright (c) 2025, Speechmatics / Cantab Research Ltd
#


from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from typing import Any

__version__ = "0.1.0"


class AgentClientMessageType(str, Enum):
    """Message types that can be sent from client to server.

    These enum values represent the different types of messages that the
    client can send to the Speechmatics RT API during a transcription session.

    Attributes:
        StartRecognition: Initiates a new transcription session with
            configuration parameters.
        AddAudio: Indicates that audio data follows (not used in message
            headers, audio is sent as binary data).
        EndOfStream: Signals that no more audio data will be sent.
        SetRecognitionConfig: Updates transcription configuration during
            an active session (advanced use).
        GetSpeakers: Internal, Speechmatics only message. Allows the client to request speaker data.

    Examples:
        >>> # Starting a recognition session
        >>> message = {
        ...     "message": AgentClientMessageType.StartRecognition,
        ...     "audio_format": audio_format.to_dict(),
        ...     "transcription_config": config.to_dict()
        ... }
        >>>
        >>> # Ending the session
        >>> end_message = {
        ...     "message": AgentClientMessageType.END_OF_STREAM,
        ...     "last_seq_no": sequence_number
        ... }
    """

    START_RECOGNITION = "StartRecognition"
    ADD_AUDIO = "AddAudio"
    END_OF_STREAM = "EndOfStream"
    SET_RECOGNITION_CONFIG = "SetRecognitionConfig"
    GET_SPEAKERS = "GetSpeakers"


class AgentServerMessageType(str, Enum):
    """Message types that can be received from the server / agent.

    These enum values represent the different types of messages that the
    Speechmatics RT API / Voice Agent SDK can send to the client.

    Attributes:
        RecognitionStarted: The recognition session has started.
        EndOfTranscript: The recognition session has ended.
        Info: Informational message.
        Warning: Warning message.
        Error: Error message.
        SpeechStarted: Speech has started.
        SpeechEnded: Speech has ended.
        TurnDetected: A turn has been detected.
        InterimSegment: An interim segment has been detected.
        FinalSegment: A final segment has been detected.
        SpeakersResult: Speakers result has been detected.

    Examples:
        >>> # Register event handlers for different message types
        >>> @client.on(AgentServerMessageType.INTERIM_SEGMENT)
        >>> def handle_interim(message):
        ...     segment: SpeakerSegment = message['segment']
        ...     print(f"Interim: {segment}")
        >>>
        >>> @client.on(AgentServerMessageType.FINAL_SEGMENT)
        >>> def handle_final(message):
        ...     segment: SpeakerSegment = message['segment']
        ...     print(f"Final: {segment}")
        >>>
        >>> @client.on(AgentServerMessageType.ERROR)
        >>> def handle_error(message):
        ...     print(f"Error: {message['reason']}")
    """

    # API messages
    RECOGNITION_STARTED = "RecognitionStarted"
    END_OF_TRANSCRIPT = "EndOfTranscript"
    INFO = "Info"
    WARNING = "Warning"
    ERROR = "Error"

    # VAD messages
    SPEECH_STARTED = "SpeechStarted"
    SPEECH_ENDED = "SpeechEnded"

    # Turn / segment messages
    TURN_DETECTED = "TurnDetected"
    INTERIM_SEGMENT = "InterimSegment"
    FINAL_SEGMENT = "FinalSegment"

    # Speaker messages
    SPEAKERS_RESULT = "SpeakersResult"


class AnnotationFlags(IntFlag):
    """Flags to apply when processing speech / objects."""

    # High-level updates
    UPDATED_NONE = auto()
    UPDATED_STRIPPED = auto()
    UPDATED_LCASE = auto()
    UPDATED_LCASE_STRIPPED = auto()
    UPDATED_FULL = auto()
    UPDATED_SPEAKERS = auto()
    UPDATED_COMPLETE = auto()
    UPDATED_LCASE_COMPLETE = auto()
    UPDATED_STRIPPED_COMPLETE = auto()
    UPDATED_LCASE_STRIPPED_COMPLETE = auto()
    UPDATED_FINALS = auto()

    # More granular details on the word content
    HAS_PARTIAL = auto()
    HAS_FINAL = auto()
    STARTS_WITH_FINAL = auto()
    ENDS_WITH_FINAL = auto()
    HAS_EOS = auto()
    ENDS_WITH_EOS = auto()
    STARTS_WITH_SB = auto()
    ENDS_WITH_SB = auto()
    HAS_SB = auto()
    ONLY_SB = auto()
    HAS_DISFLUENCY = auto()
    ENDS_WITH_DISFLUENCY = auto()
    HIGH_DISFLUENCY_COUNT = auto()
    VERY_SLOW_SPEAKER = auto()
    SLOW_SPEAKER = auto()
    FAST_SPEAKER = auto()
    ONLY_PUNCTUATION = auto()
    MULTIPLE_SPEAKERS = auto()
    LOW_CONFIDENCE_WORDS = auto()
    NO_TEXT = auto()


class EndOfUtteranceMode(str, Enum):
    """End of turn delay options for transcription."""

    NONE = "none"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class DiarizationFocusMode(str, Enum):
    """Speaker focus mode for diarization."""

    RETAIN = "retain"
    IGNORE = "ignore"


@dataclass
class AdditionalVocabEntry:
    """Additional vocabulary entry.

    Parameters:
        content: The word to add to the dictionary.
        sounds_like: Similar words to the word.
    """

    content: str
    sounds_like: list[str] = field(default_factory=list)


@dataclass
class DiarizationKnownSpeaker:
    """Known speakers for speaker diarization.

    Parameters:
        label: The label of the speaker.
        speaker_identifiers: One or more data strings for the speaker.
    """

    label: str
    speaker_identifiers: list[str]


@dataclass
class AnnotationResult:
    """Processing result."""

    flags: int = 0

    def __init__(self, *flags: AnnotationFlags):
        """Initialize the object.

        Args:
            flags: The initial flags to set.
        """
        self.flags = 0
        for flag in flags:
            self.add_flag(flag)

    def has(self, *flags: AnnotationFlags) -> bool:
        """Check if the object has all given flags."""
        return all(self.flags & flag == flag for flag in flags)

    def add(self, flag: AnnotationFlags) -> None:
        """Add a flag to the object."""
        self.flags |= flag

    def remove(self, flag: AnnotationFlags) -> None:
        """Remove a flag from the object."""
        self.flags &= ~flag

    def __str__(self):
        """String representation of the flags."""
        return f"{type(self).__name__}({', '.join(flag.name for flag in AnnotationFlags if self.has(flag))})"


@dataclass
class SpeechFragment:
    """Fragment of a speech event.

    As the transcript is processed (partials and finals), a list of SpeechFragments
    objects are accumulated and then used to form SpeechSegments objects.

    Parameters:
        start_time: Start time of the fragment in seconds (from session start).
        end_time: End time of the fragment in seconds (from session start).
        language: Language of the fragment. Defaults to `en`.
        is_eos: Whether the fragment is the end of a sentence. Defaults to `False`.
        is_final: Whether the fragment is the final fragment. Defaults to `False`.
        is_disfluency: Whether the fragment is a disfluency. Defaults to `False`.
        is_punctuation: Whether the fragment is a punctuation. Defaults to `False`.
        attaches_to: Whether the fragment attaches to the previous or next fragment (punctuation). Defaults to empty string.
        content: Content of the fragment. Defaults to empty string.
        speaker: Speaker of the fragment (if diarization is enabled). Defaults to `None`.
        confidence: Confidence of the fragment (0.0 to 1.0). Defaults to `1.0`.
        result: Raw result of the fragment from the TTS.
        annotation: Annotation for the fragment.
    """

    start_time: float
    end_time: float
    language: str = "en"
    is_eos: bool = False
    is_final: bool = False
    is_disfluency: bool = False
    is_punctuation: bool = False
    attaches_to: str = ""
    content: str = ""
    speaker: str | None = None
    confidence: float = 1.0
    result: Any | None = None
    annotation: AnnotationResult | None = None


@dataclass
class SpeakerSegment:
    """SpeechFragment items grouped by speaker_id and whether the speaker is active.

    Parameters:
        speaker_id: The ID of the speaker.
        is_active: Whether the speaker is active (emits frame).
        timestamp: The timestamp of the frame.
        language: The language of the frame.
        fragments: The list of SpeechFragment items.
        annotation: The annotation associated with the segment.
    """

    speaker_id: str | None = None
    is_active: bool = False
    timestamp: str | None = None
    language: str | None = None
    fragments: list[SpeechFragment] = field(default_factory=list)
    annotation: AnnotationResult | None = None

    def __str__(self):
        """Return a string representation of the object."""
        meta = {
            "speaker_id": self.speaker_id,
            "timestamp": self.timestamp,
            "language": self.language,
            "annotation": str(self.annotation),
            "text": self.format_text(),
        }
        return f"SpeakerSegment({', '.join(f'{k}={v}' for k, v in meta.items())})"

    def format_text(self, format: str | None = None) -> str:
        """Wrap text with speaker ID in an optional f-string format.

        Supported format variables:
            speaker_id: The ID of the speaker.
            text: The text of the fragment.
            ts: The timestamp of the fragment.
            lang: The language of the fragment.

        Args:
            format: Format to wrap the text with.

        Returns:
            str: The wrapped text.
        """
        # Cumulative contents
        content = ""

        # Assemble the text
        for frag in self.fragments:
            if content == "" or frag.attaches_to == "previous":
                content += frag.content
            else:
                content += " " + frag.content

        # Format the text, if format is provided
        if format is None or self.speaker_id is None:
            return content
        return format.format(
            **{
                "speaker_id": self.speaker_id,
                "text": content,
                "ts": self.timestamp,
                "lang": self.language,
            }
        )

    def _as_attributes(
        self, active_format: str | None = None, passive_format: str | None = None
    ) -> dict[str, Any]:
        """Return a dictionary of attributes for creating platform-specific objects.

        Args:
            active_format: Format to wrap the text with.
            passive_format: Format to wrap the text with. Defaults to `active_format`.

        Returns:
            dict[str, Any]: The dictionary of attributes.
        """
        return {
            "text": self.format_text(
                active_format if self.is_active else passive_format or active_format
            ),
            "user_id": self.speaker_id or "",
            "timestamp": self.timestamp,
            "language": self.language,
            "result": [frag.result for frag in self.fragments],
        }
