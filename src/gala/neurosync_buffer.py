"""
Processeur pour NeuroSync - Traite les chunks audio et gère les blendshapes.
"""

from __future__ import annotations

import asyncio
import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import AudioRawFrame

from .neurosync import NeuroSyncClient

@dataclass
class VisemeFrame:
    """Frame contenant un visème"""
    id: int  # ID du visème
    time_ms: int  # Timing en millisecondes

@dataclass
class BlendshapeFrame:
    """Frame contenant des blendshapes complets"""
    values: list[float]  # Tableau de valeurs (0-1) pour chaque blendshape
    frame_idx: int  # Index du frame

@dataclass
class NeuroSyncBufferConfig:
    """Configuration pour le buffer NeuroSync"""
    min_frames: int = 9  # Nombre minimum de frames à accumuler avant envoi
    sample_rate: int = 16000  # Taux d'échantillonnage en Hz
    bytes_per_frame: int = 470  # Taille d'un frame audio en octets
    debug_save: bool = False  # Sauvegarder les chunks pour debug
    log_dir: Path = Path("audio_logs")  # Dossier de logs

class NeuroSyncBufferProcessor(FrameProcessor):
    """
    Processeur qui accumule l'audio et l'envoie au serveur NeuroSync,
    puis traite les blendshapes reçus.
    """
    
    def __init__(self, client, config=None):
        super().__init__(name="neurosync_buffer")
        self.client = client
        self.config = config or NeuroSyncBufferConfig()
        self.config.log_dir.mkdir(exist_ok=True)
        
        self._buffer = bytearray()
        self._viseme_queue = []
        self._blendshape_queue = []
        
        # Configurer le callback pour les blendshapes/visèmes
        if hasattr(self.client, "set_blendshapes_callback"):
            self.client.set_blendshapes_callback(self._handle_blendshapes)
        
        # Horodatage pour générer des noms de fichiers uniques
        self._chunk_counter = 0
        
    def _handle_blendshapes(self, blendshapes):
        """
        Callback appelé quand des blendshapes sont reçus du serveur NeuroSync.
        Cette méthode est appelée depuis un thread du client, pas depuis le thread asyncio.
        """
        try:
            if not blendshapes or not isinstance(blendshapes, list):
                return
                
            # Extraire les visèmes à partir des blendshapes
            # Généralement, on utilise les valeurs de bouche pour déterminer le visème
            for i, frame in enumerate(blendshapes):
                if len(frame) < 40:  # Vérifier qu'on a assez de valeurs
                    continue
                    
                # Déterminer un ID de visème à partir des valeurs de blendshapes
                # Cette logique dépend de votre modèle spécifique - à adapter
                jaw_open = frame[17] if len(frame) > 17 else 0
                mouth_funnel = frame[19] if len(frame) > 19 else 0
                mouth_pucker = frame[20] if len(frame) > 20 else 0
                
                # Logique simplifiée pour déterminer le visème
                viseme_id = 0  # Neutre par défaut
                if jaw_open > 0.3:
                    if mouth_funnel > 0.3:
                        viseme_id = 1  # "aa"
                    elif mouth_pucker > 0.3:
                        viseme_id = 3  # "oo"
                    else:
                        viseme_id = 2  # "ah"
                        
                # Ajouter à la file d'attente pour traitement dans le thread asyncio
                time_ms = i * 33  # ~30fps = ~33ms par frame
                self._viseme_queue.append(VisemeFrame(viseme_id, time_ms))
                
                # Ajouter le frame complet de blendshapes pour LiveLink
                self._blendshape_queue.append(BlendshapeFrame(frame, i))
                
            print(f"✅ Traité {len(blendshapes)} frames de blendshapes")
                
        except Exception as e:
            print(f"❌ Erreur de traitement des blendshapes: {e}")
            
    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # Initialisation
        await super().process_frame(frame, direction)
        
        # Traiter les files d'attente de visèmes et blendshapes
        while self._viseme_queue:
            viseme_frame = self._viseme_queue.pop(0)
            await self.push_frame(viseme_frame, direction)
            
        while self._blendshape_queue:
            blendshape_frame = self._blendshape_queue.pop(0)
            await self.push_frame(blendshape_frame, direction)
            
        # Traitement de l'audio uniquement en aval (des microphones vers les haut-parleurs)
        if direction == FrameDirection.DOWNSTREAM and hasattr(frame, "audio") and frame.audio:
            audio_data = frame.audio
            if audio_data and isinstance(audio_data, bytes):
                # Ajouter au buffer
                self._buffer.extend(audio_data)
                
                # Vérifier si on a assez de données pour envoyer
                min_size = self.config.min_frames * self.config.bytes_per_frame
                if len(self._buffer) >= min_size:
                    # Envoyer les données
                    chunk_size = len(self._buffer)
                    print(f"⚡ NeuroSync: envoi de {chunk_size} octets (~{chunk_size/self.config.bytes_per_frame:.1f} frames)")
                    
                    # Debug: sauvegarder le chunk
                    if self.config.debug_save:
                        self._chunk_counter += 1
                        debug_path = self.config.log_dir / f"chunk_{self._chunk_counter:03d}.pcm"
                        with open(debug_path, "wb") as f:
                            f.write(self._buffer)
                    
                    # Envoyer de façon non-bloquante
                    asyncio.create_task(self.client.send_audio(
                        bytes(self._buffer), 
                        sample_rate=self.config.sample_rate
                    ))
                    
                    # Vider le buffer
                    self._buffer.clear()
        
        # Propager le frame
        await self.push_frame(frame, direction)
