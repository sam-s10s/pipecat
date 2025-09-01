#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics TTS service integration."""

import os
from typing import AsyncGenerator, Optional

import aiohttp
import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class SpeechmaticsTTSService(TTSService):
    """Speechmatics TTS service implementation.

    This service provides text-to-speech synthesis using the Speechmatics HTTP API.
    It converts text to speech and returns raw PCM audio data for real-time playback.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Speechmatics TTS service.

        Parameters:
            voice: Voice model to use for synthesis. Defaults to "sarah".
        """

        voice: str = "sarah"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: Optional[int] = 16000,
        params: InputParams | None = None,
        **kwargs,
    ):
        """Initialize the Speechmatics TTS service.

        Args:
            api_key: Speechmatics API key for authentication. Uses environment variable
                `SPEECHMATICS_API_KEY` if not provided.
            base_url: Base URL for Speechmatics TTS API. Defaults to
                `https://preview.tts.speechmatics.com/generate`.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            params: Optional[InputParams]: Input parameters for the service.
            **kwargs: Additional arguments passed to TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Service parameters
        self._api_key: str = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self._base_url: str = base_url or "https://preview.tts.speechmatics.com/generate"
        self._session = aiohttp_session

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")
        if not self._session:
            raise ValueError("Missing aiohttp session")

        # Default parameters
        self._params = params or SpeechmaticsTTSService.InputParams()

        # Set voice from parameters
        self.set_voice(self._params.voice)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechmatics service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Speechmatics' HTTP API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voice": self._voice_id,
        }

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_message = f"Speechmatics TTS error: HTTP {response.status}"
                    logger.error(error_message)
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                first_chunk = True
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue

                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False

                    # Calculate processing metrics for logging
                    chunk_bytes = len(chunk)
                    samples = chunk_bytes // 2  # 2 bytes per int16 sample
                    duration_seconds = samples / float(self.sample_rate)
                    logger.debug(
                        f"Processing {chunk_bytes} bytes ({samples} samples, {duration_seconds:.4f}s)"
                    )

                    # Directly yield the raw audio chunk without conversion
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

        except Exception as e:
            logger.exception(f"Error generating TTS: {e}")
            yield ErrorFrame(error=f"Speechmatics TTS error: {str(e)}")
        finally:
            yield TTSStoppedFrame()
