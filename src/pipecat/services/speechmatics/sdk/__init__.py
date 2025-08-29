#
# Copyright (c) 2025, Speechmatics / Cantab Research Ltd
#


"""Voice Agents SDK.

A comprehensive set of utility classes tailored for Voice Agents and
using the Speechmatics Python Real-Time SDK, including the processing of
partial and final transcription from the STT engine into accumulated
transcriptions with flags to indicate changes between messages, etc.
"""

from speechmatics.rt import (
    AudioEncoding,
    AudioFormat,
    OperatingPoint,
)

from ._client import VoiceAgentClient
from ._models import (
    AdditionalVocabEntry,
    AgentClientMessageType,
    AgentServerMessageType,
    AnnotationFlags,
    AnnotationResult,
    DiarizationFocusMode,
    DiarizationKnownSpeaker,
    DiarizationSpeakerConfig,
    EndOfUtteranceMode,
    SpeakerSegment,
    SpeechFragment,
    VoiceAgentConfig,
    __version__,
)

__all__ = [
    # SDK
    "__version__",
    # Conversation config
    "VoiceAgentConfig",
    "EndOfUtteranceMode",
    "DiarizationSpeakerConfig",
    "DiarizationFocusMode",
    "AdditionalVocabEntry",
    "DiarizationKnownSpeaker",
    "AudioEncoding",
    "AudioFormat",
    "OperatingPoint",
    # Transcription models
    "AnnotationFlags",
    "AnnotationResult",
    "SpeakerSegment",
    "SpeechFragment",
    # Client
    "VoiceAgentClient",
    "AgentClientMessageType",
    "AgentServerMessageType",
]
