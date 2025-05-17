"""Envoi asynchrone de l'audio vers le service NeuroSync.

Ce module fournit une classe `NeuroSyncClient` permettant d'envoyer les
segments audio générés par le TTS vers une API NeuroSync locale afin de
obtenir une synchronisation labiale.

L'implémentation utilise simplement une requête HTTP POST mais peut être
adaptée en fonction de l'API exacte. La méthode `send_audio` est conçue
pour être appelée depuis un processor du pipeline.
"""

from __future__ import annotations

import aiohttp
from dataclasses import dataclass


@dataclass
class NeuroSyncClient:
    """Client minimal pour l'API NeuroSync."""

    host: str = "127.0.0.1"
    port: int = 5005
    endpoint: str = "/audio"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.endpoint}"

    async def send_audio(
        self, audio: bytes, sample_rate: int, channels: int = 1
    ) -> None:
        """Envoie un segment audio PCM à l'API NeuroSync."""
        async with aiohttp.ClientSession() as session:
            params = {"sr": sample_rate, "ch": channels}
            try:
                await session.post(self.url, params=params, data=audio)
            except Exception as exc:  # pragma: no cover - simple log
                # En production on utiliserait loguru/logging
                print(f"NeuroSyncClient error: {exc}")


class NeuroSyncProcessor:
    """Processor pipeline qui transmet chaque segment audio à NeuroSync."""

    def __init__(self, client: NeuroSyncClient) -> None:
        self.client = client
        self.name = "neurosync_processor"

    async def process_frame(self, frame, direction):  # type: ignore[no-untyped-def]
        audio = getattr(frame, "audio", None)
        if audio is not None and hasattr(audio, "audio"):
            await self.client.send_audio(
                audio.audio, audio.sample_rate, getattr(audio, "channels", 1)
            )
        return [frame]
