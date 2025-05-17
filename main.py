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
import sys
import wave
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parent / "src"))

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
from pipecat.frames.frames import TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.stt import OpenAISTTService
from src.gala.neurosync import NeuroSyncClient, NeuroSyncProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

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

class LipSyncProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="lip_sync")

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, TTSStartedFrame):
            print("⚡ TTS started – lance l'animation")
        elif isinstance(frame, TTSStoppedFrame):
            print("✅ TTS finished – stop animation")

        # Utiliser la méthode de la classe parente au lieu de push_frame
        return await super().process_frame(frame, direction)

class AudioLoggerProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="audio_logger")
        self.log_dir = Path("audio_logs")
        self.log_dir.mkdir(exist_ok=True)
        
    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        if hasattr(frame, "audio") and frame.audio is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"tts_output_{timestamp}.wav"
            
            logger.info(f"Enregistrement audio: {len(frame.audio)} bytes → {filename}")
            
            with wave.open(str(filename), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(frame.audio)
                
        return await super().process_frame(frame, direction)

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
        model="gpt-4o-mini-tts",  # Nouveau modèle
        voice="ash",             # Voix compatible avec gpt-4o-mini-tts
        sample_rate=24000,
        response_format="pcm",
        stream=False,               # Activer le streaming
        instructions="Voix de pirate, énergique et enthousiaste"  # Instructions spécifiques
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
        audio_out_buffer_size=1024*1024,  # Buffer plus grand
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
        tts,
        AudioLoggerProcessor(),
        LipSyncProcessor(),
        NeuroSyncProcessor(NeuroSyncClient()),
        context_agg.assistant(),
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
    # Ne pas essayer de filtrer si c'est le bot ou non
    # Juste saluer tout nouveau client
    client_id = client.get("id", "inconnu")
    logger.info("Client connecté : %s", client_id)
    
    greeting = "Je suis Gala, le pirate le plus redoutable des 7 mers."
    
    assistant_ctx = context_agg.assistant()
    assistant_ctx.add_messages([{"role": "assistant", "content": greeting}])
    
    await task.queue_frames([TTSSpeakFrame(greeting)])


@transport.event_handler("on_client_disconnected")
async def on_client_disconnected(transport, client):
    logger.info("Client déconnecté : %s", client.get("id"))
    await task.cancel()


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


