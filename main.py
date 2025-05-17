#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gala – Pipeline Pipecat simplifié (Deepgram STT → Gemini (format OpenAI) → TTS → Daily).

Fonctionnalités :
- VAD Silero pour découper la parole
- Deepgram pour la reconnaissance vocale FR
- Gemini 2.0-flash-001 via l'interface OpenAI-compatible de Pipecat
- TTS OpenAI ("nova") ou ElevenLabs au choix
- Transport Daily : le bot rejoint une room et diffuse l'audio

Variables nécessaires (dans .env ou l'environnement) :
  DEEPGRAM_API_KEY
  GEMINI_API_KEY
  DAILY_ROOM_URL
Optionnels :
  OPENAI_API_KEY     (si TTS OpenAI)
  ELEVENLABS_API_KEY (si TTS ElevenLabs + TTS_SERVICE=elevenlabs)
  GEMINI_MODEL       (défaut : gemini-2.0-flash-001)
  DAILY_API_TOKEN    (si la room Daily est sécurisée)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Pipecat
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.services.google.llm_openai import GoogleLLMOpenAIBetaService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.stt import OpenAISTTService
from gala.neurosync import NeuroSyncClient, NeuroSyncProcessor

# ---------------------------------------------------------------------------
# Chargement des variables d'environnement
# ---------------------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

DEEPGRAM_API_KEY = os.getenv(
    "DEEPGRAM_API_KEY", "dbab4e489478bd4338ad6cbb3901a433550d7cf1"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
DAILY_API_TOKEN = os.getenv("DAILY_API_TOKEN")
BOT_NAME = os.getenv("BOT_NAME", "Gala")
TTS_SERVICE = os.getenv("TTS_SERVICE", "openai").lower()

REQUIRED = {
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "DAILY_ROOM_URL": DAILY_ROOM_URL,
}
for name, val in REQUIRED.items():
    if not val:
        raise SystemExit(f"[Gala] Variable d'environnement manquante : {name}")

if TTS_SERVICE == "elevenlabs" and not ELEVENLABS_API_KEY:
    raise SystemExit("[Gala] ELEVENLABS_API_KEY requis si TTS_SERVICE=elevenlabs")


# ---------------------------------------------------------------------------
# Définition d'une fonction « get_current_weather » (exemple function-calling)
# ---------------------------------------------------------------------------
async def fetch_weather_from_api(params: FunctionCallParams):
    """Simule un appel météo puis renvoie un résultat."""
    await params.llm.push_frame(TTSSpeakFrame("Je vérifie la météo pour vous."))
    await asyncio.sleep(0.5)
    await params.result_callback({"conditions": "ensoleillé", "temperature": "24"})


weather_schema = FunctionSchema(
    name="get_current_weather",
    description="Obtenir la météo actuelle pour une ville donnée",
    properties={
        "location": {
            "type": "string",
            "description": "Ville et pays, ex. : Paris, France",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Unité de température, déduite de la localisation utilisateur.",
        },
    },
    required=["location", "format"],
)

# ---------------------------------------------------------------------------
# Services Pipecat (STT, LLM, TTS)
# ---------------------------------------------------------------------------

## STT Deepgram
# stt = DeepgramSTTService(
#    api_key       = DEEPGRAM_API_KEY,
#    language      = "fr",
#    model         = "2-general",
#    vad           = False,          # ← plus de double VAD
#    punctuate     = True,
#    interim_results = False,
#    smart_format  = True,
# )

stt = OpenAISTTService(
    api_key=OPENAI_API_KEY,  # déjà dans ton .env
    language="fr",  # optionnel, sinon auto-detect
    model="whisper-1",  # modèle public actuel
    vad=False,  # on garde *un seul* VAD = Silero
)


# LLM Gemini (interface OpenAI)
llm = GoogleLLMOpenAIBetaService(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
llm.register_function("get_current_weather", fetch_weather_from_api)

# Choix du TTS
if TTS_SERVICE == "elevenlabs":
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY, voice_id="pNInz6obpgDQGcFmaJgB"
    )

if TTS_SERVICE == "openai":
    tts = OpenAITTSService(
        api_key=OPENAI_API_KEY,
        model="tts-1",  # ou "tts-1-hd"
        voice="nova",
        sample_rate=24000,
        response_format="wav",  # utile pour l’inspection
        stream=False,  # force la réponse en une seule partie
    )


# ---------------------------------------------------------------------------
# Transport Daily + VAD Silero
# ---------------------------------------------------------------------------
transport = DailyTransport(
    room_url=DAILY_ROOM_URL,
    token=DAILY_API_TOKEN,
    bot_name=BOT_NAME,
    params=DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(confidence=0.6, start_secs=0.1, stop_secs=0.7)
        ),
    ),
)

# Buffer audio (facultatif, pour exporter les tours)
buffer = AudioBufferProcessor(num_channels=1, enable_turn_audio=True)

# ---------------------------------------------------------------------------
# Contexte LLM
# ---------------------------------------------------------------------------
messages = [
    {
        "role": "system",
        "content": "tu es un pirate du nom de Gala, tu as toujours des histoires à raconter, tu es très amical et tu aimes parler de tes aventures .",
    },
]
context = OpenAILLMContext(messages, ToolsSchema(standard_tools=[weather_schema]))
context_agg = llm.create_context_aggregator(context)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
# Pipeline final
pipeline = Pipeline(
    [
        transport.input(),
        stt,
        context_agg.user(),
        llm,
        tts,  # 1. synthèse
        NeuroSyncProcessor(NeuroSyncClient()),  # envoi audio vers NeuroSync
        context_agg.assistant(),  # 2. archivage de la réponse
        transport.output(),
        buffer,
    ]
)

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
)


# ---------------------------------------------------------------------------
# Handlers Daily
# ---------------------------------------------------------------------------
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    if client.get("id") != transport.bot_id:
        logger.info("Client connecté : %s", client.get("id"))

        greeting = "Je suis Gala, le pirate le plus redoutable des 7 mers."

        assistant_ctx = context_agg.assistant()
        assistant_ctx.add_messages([{"role": "assistant", "content": greeting}])

        # On fait parler Gala mais on n'appelle pas Gemini tout de suite → pas de risque d'erreur 400
        await task.queue_frames(
            [
                TTSSpeakFrame(greeting),
            ]
        )


@transport.event_handler("on_client_disconnected")
async def on_client_disconnected(transport, client):
    logger.info("Client déconnecté : %s", client.get("id"))
    await task.cancel()


# ─── callbacks synchro labiale ───────────────────────────────────
# Ces callbacks déclenchent simplement le début et la fin de l'animation faciale
# pendant que le processor ``NeuroSyncProcessor`` envoie l'audio en continu.
@tts.event_handler("on_tts_started")
async def _(*_):
    print("⚡ TTS started – lance l’animation")


@tts.event_handler("on_tts_finished")
async def _(*_):
    print("✅ TTS finished – stop animation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    logger.info("Gala • démarrage du pipeline – salle %s", DAILY_ROOM_URL)
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt par l'utilisateur")
