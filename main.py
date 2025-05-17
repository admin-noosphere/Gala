#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gala – pipeline Pipecat (Deepgram STT ➜ OpenAI LLM ➜ ElevenLabs/OpenAI TTS)
avec export audio complet vers NeuroSync.
Compatible Pipecat 0.0.68.dev11 (API Runner/Task).
"""
from __future__ import annotations

import asyncio
import io
import os
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import sys
import logging
import inspect

# Ajout du chemin de Pipecat local au PYTHONPATH
local_path = Path(__file__).parent / "vendor" / "pipecat" / "src"
sys.path.insert(0, str(local_path))

# Vérification du chemin Python pour débogage
print("Chemins Python:")
for p in sys.path:
    print(f"  - {p}")

import aiohttp
from dotenv import load_dotenv
from loguru import logger

# Configuration du logger pour plus de détails sur le VAD et l'audio
logger.remove()
logger.add(sys.stderr, level="DEBUG")  # Changez à DEBUG pour tout voir

# Pipecat imports – STT/LLM/TTS/transport
from pipecat.processors.frame_processor import FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.stt_service import STTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    TextFrame, 
    LLMTextFrame, 
    TTSTextFrame, 
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame
)

# Désactiver tous les logs de websockets et HTTP
logging.getLogger('websockets').setLevel(logging.CRITICAL)
logging.getLogger('httpcore').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('openai').setLevel(logging.WARNING)

# Configuration du logger Pipecat
import sys
from loguru import logger

# Fonction utilitaire pour afficher les détails des frames
def log_frame_details(frame, source):
    """Affiche les détails d'une frame pour le débogage."""
    try:
        frame_type = type(frame).__name__
        details = []
        
        # Ajouter les attributs communs
        if hasattr(frame, 'text'):
            details.append(f"text: '{frame.text}'")
        if hasattr(frame, 'audio') and frame.audio:
            details.append(f"audio: {len(frame.audio)} octets")
        if hasattr(frame, 'sample_rate'):
            details.append(f"sample_rate: {frame.sample_rate}Hz")
        if hasattr(frame, '_direction'):
            details.append(f"direction: {frame._direction}")
            
        # Afficher les détails
        if details:
            logger.info(f"{source}: {frame_type} - {', '.join(details)}")
        else:
            logger.info(f"{source}: {frame_type} - pas d'attributs spécifiques")
    except Exception as e:
        logger.error(f"{source}: Erreur lors de l'affichage des détails de la frame: {e}")

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def pcm_to_wav_bytes(pcm: bytes, sr: int, channels: int = 1) -> bytes:
    """Encapsule du PCM 16-bit little-endian en WAV."""
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        return buf.getvalue()


class NeuroSyncClient:
    """Client HTTP minimaliste pour NeuroSync."""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def lipsync(self, wav: bytes) -> bool:
        if not self._session:
            raise RuntimeError("NeuroSync session not initialised (use async with).")
        try:
            async with self._session.post(
                self.api_url,
                data=wav,
                headers={"Content-Type": "audio/wav"},
                timeout=15,
            ) as resp:
                if resp.status == 200:
                    logger.debug("NeuroSync OK ({} bytes)", len(wav))
                    return True
                logger.error("NeuroSync {} – {}", resp.status, await resp.text())
                return False
        except Exception as exc:
            logger.exception("NeuroSync request failed: {}", exc)
            return False


# ---------------------------------------------------------------------------
# Chargement des variables d'environnement
# ---------------------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL")
BOT_NAME = os.getenv("BOT_NAME", "Gala")
NEUROSYNC_API_URL = os.getenv("NEUROSYNC_API_URL", "http://localhost:8000/lipsync")
DAILY_API_TOKEN = os.getenv("DAILY_TOKEN")
# Choix du service TTS (elevenlabs ou openai)
TTS_SERVICE = os.getenv("TTS_SERVICE", "openai").lower()

for var_name, var_val in {
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "DAILY_ROOM_URL": DAILY_ROOM_URL,
    #"DAILY_API_TOKEN": DAILY_API_TOKEN,
}.items():
    print(var_name, var_val)
    if not var_val:
        logger.error("La variable {} est manquante dans .env", var_name)
        raise SystemExit(1)

# Vérification conditionnelle pour ElevenLabs
if TTS_SERVICE == "elevenlabs" and not ELEVENLABS_API_KEY:
    logger.error("La variable ELEVENLABS_API_KEY est manquante dans .env (requise pour TTS_SERVICE=elevenlabs)")
    raise SystemExit(1)



# ---------------------------------------------------------------------------
# Services Pipecat
# ---------------------------------------------------------------------------

# Création d'un gestionnaire d'événements pour Deepgram
def deepgram_event_handler(event_name):
    async def handler(*args, **kwargs):
        logger.info(f"Deepgram event {event_name}: {args}, {kwargs}")
    return handler

# Configuration du service STT Deepgram avec plus de logging
dg_stt = DeepgramSTTService(
    api_key=DEEPGRAM_API_KEY,
    language="fr", tier="nova",
    model="2-general",
    vad=True,
    punctuate=True,
    diarize=False,
    audio_passthrough=True,
    interim_results=True,
    smart_format=True
)

# Ajout des gestionnaires d'événements pour Deepgram
dg_stt._register_event_handler("on_speech_started")
dg_stt._register_event_handler("on_utterance_end")
dg_stt._register_event_handler("on_transcript_received")
dg_stt._register_event_handler("on_audio_received")
dg_stt._register_event_handler("process_frame")

@dg_stt.event_handler("process_frame")
async def on_dg_process_frame(processor, frame):
    logger.info(f"Deepgram: Processing frame - {type(frame).__name__}")
    if isinstance(frame, TranscriptionFrame):
        logger.info(f"Deepgram: Transcription frame - '{frame.text}'")

@dg_stt.event_handler("on_speech_started")
async def on_speech_started(*args, **kwargs):
    logger.info(f"Deepgram: Détection de parole - {args}, {kwargs}")

@dg_stt.event_handler("on_utterance_end")
async def on_utterance_end(*args, **kwargs):
    logger.info(f"Deepgram: Fin d'énoncé - {args}, {kwargs}")

@dg_stt.event_handler("on_transcript_received")
async def on_transcript_received(*args, **kwargs):
    transcript = kwargs.get('transcript', '')
    logger.info(f"Deepgram: Transcription reçue - '{transcript}'")
    if not transcript:
        logger.warning("Deepgram: Transcription vide reçue!")
    else:
        logger.info(f"Deepgram: Transcription non vide: '{transcript}'")
    
    # Afficher plus de détails sur le contenu de kwargs
    logger.info(f"Deepgram: Détails complets - {kwargs}")

@dg_stt.event_handler("on_audio_received")
async def on_audio_received(*args, **kwargs):
    audio_data = kwargs.get('audio_data', b'')
    logger.info(f"Deepgram: Audio reçu - {len(audio_data)} octets")
    if len(audio_data) > 0:
        logger.info("Audio non vide reçu par Deepgram!")

import types

# Créer un objet SimpleNamespace complet avec tous les attributs nécessaires
llm_params = types.SimpleNamespace(
    temperature=0.7,
    max_tokens=4096,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    seed=None,
    extra={},
    max_completion_tokens=None
)

# Utilisation avec OpenAILLMService
llm = OpenAILLMService(
    api_key=OPENAI_API_KEY,
    model="gpt-4o",
    params=llm_params
)

# Sélection du service TTS selon la configuration
if TTS_SERVICE == "elevenlabs":
    # Service TTS ElevenLabs
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id="pNInz6obpgDQGcFmaJgB",
    )
else:
    # Service TTS OpenAI (par défaut)
    tts = OpenAITTSService(
        api_key=OPENAI_API_KEY,
        model="tts-1",
        voice="nova",
        sample_rate=24000,
        response_format="pcm",
        instructions="Ton: Naturel et bienveillant. Émotion: Engagement et attention. Débit: Fluide avec des pauses naturelles."
    )

# Configuration du transport Daily avec plus de logging
transport = DailyTransport(
    room_url=DAILY_ROOM_URL,
    token=DAILY_API_TOKEN,
    bot_name=BOT_NAME,
    params=DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
        audio_out_volume=1.0,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(
            confidence=0.6,
            start_secs=0.1,
            stop_secs=0.7
        )),
        transcription_enabled=True
    )
)
logger.info("Silero VAD initialisé avec les paramètres: confidence=0.6, start_secs=0.1, stop_secs=0.7")

# Enregistrer les événements personnalisés
transport._register_event_handler("on_vad_state_changed")
transport._register_event_handler("on_audio_data")
transport._register_event_handler("on_transcription")
transport._register_event_handler("on_audio_received")

@transport.event_handler("on_audio_received")
async def on_transport_audio_received(*args, **kwargs):
    logger.info(f"Daily: Audio reçu - {len(kwargs.get('audio_data', b''))} octets")

@transport.event_handler("on_participant_joined")
async def on_participant_joined(_, participant):
    logger.info(f"Daily: Participant rejoint - {participant}")
    # Capture explicite de la transcription pour ce participant
    await transport.capture_participant_transcription(participant["id"])

@transport.event_handler("on_vad_state_changed")
async def on_vad_state_changed(_, state, confidence):
    logger.info(f"Silero VAD: État changé - {state}, confiance: {confidence:.2f}")

@transport.event_handler("on_audio_data")
async def on_audio_data(_, audio_data):
    # Calculer le volume audio pour le débogage
    if hasattr(audio_data, 'data') and len(audio_data.data) > 0:
        import numpy as np
        samples = np.frombuffer(audio_data.data, dtype=np.int16)
        volume = np.abs(samples).mean()
        # Toujours afficher le volume pour déboguer
        logger.info(f"Audio: Volume détecté - {volume:.2f}")
        # Log plus détaillé pour les volumes significatifs
        if volume > 500:
            logger.info(f"Audio: Volume significatif détecté - {volume:.2f}")

@transport.event_handler("on_transcription")
async def on_transcription(_, transcript):
    logger.info(f"Daily: Transcription reçue - {transcript}")

buffer = AudioBufferProcessor(num_channels=1, enable_turn_audio=True)

@buffer.event_handler("on_bot_turn_audio_data")
async def _on_bot_turn(buf, audio: bytes, sr: int, ch: int):
    wav = pcm_to_wav_bytes(audio, sr, ch)
    logger.debug(f"NeuroSync: Envoi de {len(wav)} octets d'audio")
    await _neurosync.lipsync(wav)

@buffer.event_handler("on_user_turn_audio_data")
async def _on_user_turn(buf, audio: bytes, sr: int, ch: int):
    logger.debug(f"Utilisateur: Audio reçu - {len(audio)} octets, {sr}Hz, {ch} canaux")

# Créer un contexte de conversation
messages = [
    {
        "role": "system",
        "content": "Vous êtes un assistant IA serviable et amical. Répondez de manière concise et naturelle."
    }
]
context = OpenAILLMContext(messages)
context_aggregator = llm.create_context_aggregator(context)

# Enregistrer des événements pour les agrégateurs de contexte
context_aggregator.user()._register_event_handler("process_frame")
context_aggregator.assistant()._register_event_handler("process_frame")

@context_aggregator.user().event_handler("process_frame")
async def on_user_context_process_frame(processor, frame):
    logger.info(f"UserContext: Processing frame - {type(frame).__name__}")
    if isinstance(frame, TextFrame):
        logger.info(f"UserContext: Text frame - '{frame.text}'")

@context_aggregator.assistant().event_handler("process_frame")
async def on_assistant_context_process_frame(processor, frame):
    logger.info(f"AssistantContext: Processing frame - {type(frame).__name__}")
    if isinstance(frame, TextFrame):
        logger.info(f"AssistantContext: Text frame - '{frame.text}'")
    # Ajoutez un log pour voir la direction de la frame
    if hasattr(frame, '_direction'):
        logger.info(f"AssistantContext: Frame direction - {frame._direction}")

# Modifier le pipeline pour inclure le context_aggregator
pipeline = Pipeline([
    transport.input(),
    dg_stt,
    context_aggregator.user(),
    llm,
    context_aggregator.assistant(),
    tts,
    transport.output(),
    buffer,
])

# Créer la tâche avec des paramètres spécifiques
task = PipelineTask(
    pipeline,
    params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,
        enable_usage_metrics=True,
    )
)

# Ajoutez un gestionnaire d'événements pour le TTS
tts._register_event_handler("on_tts_start")
tts._register_event_handler("on_tts_end")
tts._register_event_handler("process_frame")
tts._register_event_handler("on_tts_data")
tts._register_event_handler("on_error")

@tts.event_handler("on_tts_start")
async def on_tts_start(*args, **kwargs):
    logger.info(f"TTS: Début de la synthèse vocale")

@tts.event_handler("on_tts_end")
async def on_tts_end(*args, **kwargs):
    logger.info(f"TTS: Fin de la synthèse vocale - {kwargs.get('audio_length', 0)} octets générés")

@tts.event_handler("process_frame")
async def on_tts_process_frame(processor, frame):
    try:
        frame_type = type(frame).__name__
        logger.debug(f"TTS: Traitement d'une frame - {frame_type}")
        log_frame_details(frame, "TTS")
        if isinstance(frame, TextFrame):
            logger.info(f"TTS: Texte reçu pour synthèse - '{frame.text}'")
        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.info(f"TTS: Fin de réponse LLM reçue")
        elif isinstance(frame, TTSAudioRawFrame) and frame.audio:
            logger.info(f"TTS: Audio frame - {len(frame.audio)} octets, {frame.sample_rate}Hz")
        elif isinstance(frame, TTSStartedFrame):
            logger.info(f"TTS: Début de synthèse vocale")
        elif isinstance(frame, TTSStoppedFrame):
            logger.info(f"TTS: Fin de synthèse vocale")
    except Exception as e:
        logger.error(f"TTS: Erreur lors du traitement de la frame: {e}")

@tts.event_handler("on_tts_data")
async def on_tts_data(_, audio_data, sample_rate, num_channels):
    logger.info(f"TTS: Données audio générées - {len(audio_data)} octets, {sample_rate}Hz, {num_channels} canaux")
    # Si vous recevez des données audio mais qu'elles ne sont pas envoyées, le problème est dans le transport

@tts.event_handler("on_error")
async def on_tts_error(_, error):
    logger.error(f"TTS service error: {error}")

# Garder uniquement le gestionnaire d'événement
@transport.event_handler("on_error")
async def on_transport_error(_, err):
    logger.error(f"Daily transport error: {err}")

# Ajoutez des gestionnaires pour le LLM
llm._register_event_handler("on_llm_response")
llm._register_event_handler("process_frame")

@llm.event_handler("process_frame")
async def on_llm_process_frame(processor, frame):
    logger.info(f"LLM: Traitement d'une frame - {type(frame).__name__}")
    log_frame_details(frame, "LLM")
    if isinstance(frame, TextFrame):
        logger.info(f"LLM: Texte reçu - '{frame.text}'")
    elif isinstance(frame, LLMFullResponseStartFrame):
        logger.info("LLM: Début de génération de réponse")
    elif isinstance(frame, LLMFullResponseEndFrame):
        logger.info("LLM: Fin de génération de réponse")

@llm.event_handler("on_llm_response")
async def on_llm_response(*args, **kwargs):
    response = kwargs.get('response', '')
    logger.info(f"LLM: Réponse générée - '{response}'")
    logger.info(f"LLM: Détails complets - {kwargs}")
    # Log la réponse complète pour débogage
    if 'completion' in kwargs:
        completion = kwargs.get('completion')
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            choice = completion.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                logger.info(f"LLM: Message complet - {message}")

# Modifier la fonction test_tts pour utiliser la file d'attente de tâches
async def test_tts():
    """Test manuel du TTS pour vérifier qu'il fonctionne correctement"""
    logger.info("Test manuel du TTS...")
    test_frame = TTSSpeakFrame(text="Ceci est un test du service TTS.")
    await task.queue_frame(test_frame)
    logger.info("Test TTS terminé.")

async def main():
    global _neurosync
    _neurosync = NeuroSyncClient(NEUROSYNC_API_URL)
    
    # Démarrer le pipeline
    logger.info(f"Gala : pipeline démarré – room {DAILY_ROOM_URL}")
    
    # Test manuel du TTS après 5 secondes
    task_test = asyncio.create_task(asyncio.sleep(5))
    await task_test
    await test_tts()
    
    # Démarrer explicitement la conversation
    # Utilisez get_context_frame() au lieu de create_text_frame
    await task.queue_frames([context_aggregator.user().get_context_frame()])
    
    # Lancement runner Pipecat
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt par l'utilisateur")