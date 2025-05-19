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
import logging

# Configuration du logger global pour capturer toute la sortie console
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"gala_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Garde aussi l'affichage sur la console
    ]
)

# Classe pour rediriger stdout et stderr vers le logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = []

    def write(self, message):
        if message.strip():  # Ignorer les lignes vides
            # Accumuler dans le tampon jusqu'à une nouvelle ligne
            if '\n' in message:
                parts = message.split('\n')
                # Traiter les parties complètes
                for part in parts[:-1]:
                    if self.buffer:
                        self.buffer.append(part)
                        logging.log(self.level, ''.join(self.buffer))
                        self.buffer.clear()
                    elif part:
                        logging.log(self.level, part)
                # Stocker la dernière partie si non vide
                if parts[-1]:
                    self.buffer.append(parts[-1])
            else:
                self.buffer.append(message)
    
    def flush(self):
        if self.buffer:
            logging.log(self.level, ''.join(self.buffer))
            self.buffer.clear()

# Rediriger stdout et stderr vers notre logger
sys.stdout = LoggerWriter(logging.INFO)
sys.stderr = LoggerWriter(logging.ERROR)

logging.info("=== Démarrage de Gala ===")

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
from pipecat.frames.frames import TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.stt import OpenAISTTService
from src.gala.neurosync import NeuroSyncClient
from src.gala.neurosync_buffer import NeuroSyncBufferProcessor, NeuroSyncBufferConfig
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
import itertools
from src.gala.livelink import LiveLinkDataTrack
from src.gala.neurosync_buffer import VisemeFrame, BlendshapeFrame

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
NS_HOST = os.getenv("NS_HOST", "127.0.0.1")
NS_PORT = int(os.getenv("NS_PORT", "6969"))
NS_BUFFER_MS = int(os.getenv("NS_BUFFER_MS", "100"))
NS_EXTRA_DELAY_MS = int(os.getenv("NS_EXTRA_DELAY_MS", "50"))

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

        # La méthode correcte à appeler
        return await super().process_frame(frame, direction)


class UtteranceRecorder(FrameProcessor):
    def __init__(self, log_dir: Path = Path("audio_logs")):
        super().__init__(name="utterance_recorder")
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self._buffer = bytearray()
        self._recording = False
        self._counter = itertools.count(1)
        self._debug_counter = 0
        logger.info(f"UtteranceRecorder initialisé - dossier: {self.log_dir}")
        
        # Sauvegarde de tous les types de frames pour déboguer
        self._save_all_frames = False

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # --- (0) Toujours appeler super() en premier ---
        await super().process_frame(frame, direction)
        
        # --- (1) Traitement et enregistrement ---
        # Ne logge que les frames importants, pas les frames audio bruts
        if not isinstance(frame, TTSAudioRawFrame) and not isinstance(frame, VisemeFrame) and not isinstance(frame, BlendshapeFrame):
            logger.info(f"Frame reçu: {type(frame).__name__}")

        # Capture TTSStartedFrame
        if isinstance(frame, TTSStartedFrame):
            logger.info("⏺️ TTSStartedFrame détecté - début d'enregistrement")
            self._buffer.clear()
            self._recording = True

        # Capture TTSAudioRawFrame et tout frame qui pourrait contenir de l'audio
        elif hasattr(frame, "audio") and frame.audio:
            audio_data = frame.audio
            if isinstance(audio_data, bytes) and len(audio_data) > 0:
                self._debug_counter += 1
                
                # Ajouter au buffer principal si on enregistre (sans logger chaque chunk)
                if self._recording:
                    self._buffer.extend(audio_data)
                    # Log uniquement tous les 30 chunks pour réduire le bruit
                    if self._debug_counter % 30 == 0:
                        logger.info(f"📥 Buffer audio: {len(self._buffer)} octets accumulés")

        # Capture TTSStoppedFrame
        elif isinstance(frame, TTSStoppedFrame):
            logger.info("⏹️ TTSStoppedFrame détecté - fin d'enregistrement")
            if self._recording and len(self._buffer) > 0:
                idx = next(self._counter)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = self.log_dir / f"utterance_{idx:03d}_{ts}.wav"

                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(self._buffer)

                logger.info(f"✅ Utterance complète enregistrée: {wav_path} ({len(self._buffer)} octets)")
            else:
                logger.warning("⚠️ Fin d'enregistrement mais buffer vide ou pas d'enregistrement en cours")

            self._recording = False
            
        # --- (2) Toujours propager la frame à la fin ---
        await self.push_frame(frame, direction)


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
        voice="ash",  # Voix compatible avec gpt-4o-mini-tts
        sample_rate=24000,
        response_format="wav",
        stream=True,  # Activer le streaming
        instructions="Voix de pirate, énergique et enthousiaste",  # Instructions spécifiques
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
        audio_out_buffer_size=1024 * 1024,  # Buffer plus grand
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
        "content": """
        Personality and Tone
Identity
Tu incarnes Gala "la Tempête Écarlate", célèbre capitaine corsaire des Sept Mers. Ancien mousse devenu légende, Gala a vu plus d'horizons qu'il n'existe d'étoiles sur la voûte céleste ; il vogue maintenant dans les eaux numériques pour partager ses récits, transmettre son savoir de vieux loup de mer et garder l'esprit d'aventure bien vivant.

Task
Être un guide et un compagnon bavard :

raconter des histoires de piraterie,

aider l'utilisateur dans ses quêtes (infos, conseils, inspiration),

toujours maintenir l'ambiance maritime et intrépide,

respecter les règles de confirmation orthographique lorsque l'utilisateur fournit des noms, numéros ou tout détail sensible.

Demeanor
Chaleureux, bravache, légèrement espiègle ; jamais condescendant. Gala accueille chaque échange comme un nouveau port à explorer.

Tone
Langage coloré, truffé d'expressions marines : « Ahoy ! », « Par tous les flibustiers ! », « Hissez haut ! ». Reste néanmoins clair et compréhensible.

Level of Enthusiasm
Élevé : l'énergie d'un capitaine qui hisse la grand-voile face au vent.

Level of Formality
Plutôt décontracté ; tutoiement chaleureux. Mais sait passer au vouvoiement respectueux si le contexte l'exige.

Level of Emotion
Expressif : rires francs, étonnements théâtraux, compassion sincère quand nécessaire.

Filler Words
Occasionnellement—des interjections pirates : « Arr ! », « Ho ho ! », « Par la barbe de Barbe-Noire ! ».

Pacing
Rythme vif comme des vagues sous le vent, mais sait ralentir pour détailler un récit ou une explication complexe.

Other details
Garde une boussole imaginaire qu'il consulte avant de donner des directives (« Un coup d'œil à ma boussole intérieure… »).

Aime ponctuer ses histoires d'une morale ou d'un trésor de sagesse.

Jamais vulgaire ; la truculence doit rester bon enfant.

Rappelle parfois sa devise : « Libre comme l'écume, fidèle comme la marée. »

""",
    },
]
context = OpenAILLMContext(messages, ToolsSchema(standard_tools=[weather_schema]))
context_agg = llm.create_context_aggregator(context)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
# Initialiser les services NeuroSync
neurosync_client = NeuroSyncClient(
    host=NS_HOST,
    port=NS_PORT
)

# Initialiser LiveLink
livelink_processor = LiveLinkDataTrack(transport, use_udp=True, udp_ip="192.168.1.14")

# Initialiser NeuroSync avec une référence à LiveLink
neurosync_proc = NeuroSyncBufferProcessor(
    neurosync_client,
    livelink_processor=livelink_processor,
    config=NeuroSyncBufferConfig(
        min_frames=9,
        sample_rate=16000,
        bytes_per_frame=470,
        debug_save=True
    )
)

# Pipeline - Retirer LiveLinkDataTrack de la chaîne principale
pipeline = Pipeline(
    [
        transport.input(),
        stt,
        context_agg.user(),
        llm,
        tts,
        neurosync_proc, # Should be active to process TTS audio and produce blendshapes
        #LiveLinkDataTrack(transport, use_udp=True, udp_ip="192.168.1.32"), # Consumes frames from neurosync_proc
        context_agg.assistant(),
        UtteranceRecorder(),
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
