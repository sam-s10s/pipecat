#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics text-to-speech service implementation.

This module provides HTTP-based text-to-speech services using Speechmatics' API
for audio synthesis.
"""

from typing import AsyncGenerator, Optional

import aiohttp
import numpy as np
from loguru import logger

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
    """Speechmatics HTTP-based text-to-speech service.

    Provides text-to-speech synthesis using Speechmatics' HTTP API for batch processing.
    The service converts text to speech and returns raw PCM audio data.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "sarah",
        aiohttp_session: aiohttp.ClientSession,
        url: str = "https://preview.tts.speechmatics.com/generate",
        sample_rate: Optional[int] = 16000,
        **kwargs,
    ):
        """Initialize Speechmatics TTS service.

        Args:
            api_key: Speechmatics API key for authentication.
            voice: Voice model to use for synthesis. Defaults to "gwen".
            aiohttp_session: Shared aiohttp session for HTTP requests.
            url: Speechmatics TTS API endpoint.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._session = aiohttp_session
        self._url = url
        self.set_voice(voice)

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

            async with self._session.post(self._url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_message = f"Speechmatics TTS error: HTTP {response.status}"
                    logger.error(error_message)
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                # Process the response in streaming chunks
                first_chunk = True
                buffer = b""

                # Helper to convert all complete 4-byte float samples from buffer into one frame
                def _emit_complete_samples():
                    nonlocal buffer
                    if len(buffer) < 4:
                        return None
                    complete_samples = len(buffer) // 4
                    complete_bytes = complete_samples * 4
                    # Log what we're about to process
                    duration_seconds = complete_samples / float(self.sample_rate)
                    logger.debug(
                        f"Processing {complete_bytes} bytes ({complete_samples} samples, {duration_seconds:.4f}s)"
                    )
                    # Convert raw bytes to numpy array of 32-bit floats (little-endian)
                    float_samples = np.frombuffer(buffer[:complete_bytes], dtype="<f4")
                    # Convert float samples (-1.0 to 1.0) to 16-bit integers
                    int16_samples = (float_samples * 32767).astype(np.int16)
                    # Keep remaining bytes for next iteration
                    buffer = buffer[complete_bytes:]
                    return TTSAudioRawFrame(
                        audio=int16_samples.tobytes(),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue

                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False

                    buffer += chunk
                    logger.debug(f"Received chunk: {len(chunk)} bytes")

                    # Warn if chunk size is not divisible by 4 bytes (float32 sample size)
                    if len(chunk) % 4 != 0:
                        logger.warning(
                            f"Received chunk of {len(chunk)} bytes, not divisible by 4 (float32 sample size)"
                        )

                    # Emit a frame for all complete samples currently in buffer
                    frame = _emit_complete_samples()
                    if frame:
                        yield frame

                # Process any remaining bytes in buffer after streaming ends
                frame = _emit_complete_samples()
                if frame:
                    yield frame

        except Exception as e:
            logger.exception(f"Error generating TTS: {e}")
            yield ErrorFrame(error=f"Speechmatics TTS error: {str(e)}")
        finally:
            yield TTSStoppedFrame()
