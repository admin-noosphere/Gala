from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from .neurosync import NeuroSyncClient


@dataclass
class NeuroSyncBufferConfig:
    """Configuration for buffered sending to NeuroSync."""

    buffer_ms: int = 100
    extra_delay_ms: int = 50


class NeuroSyncBufferProcessor(FrameProcessor):
    """Buffer audio frames before sending them to NeuroSync.

    The buffered audio is also forwarded downstream after ``buffer_ms`` plus
    ``extra_delay_ms`` to simulate processing latency.
    """

    def __init__(
        self, client: NeuroSyncClient, config: Optional[NeuroSyncBufferConfig] = None
    ):
        super().__init__(name="neurosync_buffer")
        self.client = client
        self.config = config or NeuroSyncBufferConfig()
        self._buffer = bytearray()
        self._last_sent = asyncio.get_event_loop().time()

    async def _flush(self, sample_rate: int, channels: int = 1) -> None:
        if not self._buffer:
            return
        data = bytes(self._buffer)
        self._buffer.clear()
        await self.client.send_audio(data, sample_rate, channels)
        self._last_sent = asyncio.get_event_loop().time()

    async def process_frame(
        self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        audio = getattr(frame, "audio", None)
        if audio and hasattr(audio, "audio"):
            self._buffer.extend(audio.audio)
            now = asyncio.get_event_loop().time()
            if (now - self._last_sent) * 1000 >= self.config.buffer_ms:
                await self._flush(audio.sample_rate, getattr(audio, "channels", 1))

            await asyncio.sleep(
                (self.config.buffer_ms + self.config.extra_delay_ms) / 1000
            )

        if getattr(frame, "is_end", False) and self._buffer:
            await self._flush(
                audio.sample_rate if audio else 24000,
                getattr(audio, "channels", 1) if audio else 1,
            )

        return await super().process_frame(frame, direction)
