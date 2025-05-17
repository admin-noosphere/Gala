"""
Processeur pour envoyer les visèmes et blendshapes via LiveLink.
"""

from __future__ import annotations

import os
import socket
import json
import uuid
from pathlib import Path
from datetime import datetime
import struct
import numpy as np
import shutil
import logging

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from src.gala.neurosync_buffer import VisemeFrame, BlendshapeFrame


# Enum pour les blendshapes (simplifié)
class FaceBlendShape:
    """Enum pour les blendshapes LiveLink"""

    # Eyes
    EyeBlinkLeft = 0
    EyeBlinkRight = 7

    # Jaw and mouth
    JawOpen = 17
    MouthClose = 18
    MouthFunnel = 19
    MouthPucker = 20
    MouthLeft = 21
    MouthRight = 22
    MouthSmileLeft = 23
    MouthSmileRight = 24
    MouthFrownLeft = 25
    MouthFrownRight = 26
    MouthDimpleLeft = 27
    MouthDimpleRight = 28
    MouthStretchLeft = 29
    MouthStretchRight = 30
    MouthRollLower = 31
    MouthRollUpper = 32


class LiveLinkFace:
    """Encodeur du protocole LiveLink face"""

    def __init__(self, name="GalaBot", fps=60):
        # Identifiant unique du sujet LiveLink
        self.uuid = str(uuid.uuid1())
        self.name = name
        self.fps = fps
        self._version = 6
        self._blend_shapes = [0.0] * 61
        self._denominator = int(self.fps / 60)

    def set_blendshape(self, index, value):
        """Set a blendshape value (0.0-1.0)"""
        self._blend_shapes[index] = max(0.0, min(1.0, value))

    def set_blendshapes(self, values):
        """Set all blendshapes from a list of values"""
        for i, value in enumerate(values):
            if i < len(self._blend_shapes):
                self._blend_shapes[i] = max(0.0, min(1.0, value))

    def reset_blendshapes(self):
        """Reset all blendshapes to 0"""
        self._blend_shapes = [0.0] * 61

    def encode(self):
        """Encode blendshapes to LiveLink binary format"""
        # Format header
        version_packed = struct.pack("<I", self._version)
        uuid_packed = self.uuid.encode("utf-8")
        name_packed = self.name.encode("utf-8")
        name_length_packed = struct.pack("!i", len(self.name))

        # Timecode (simplified)
        now = datetime.now()
        frames = int((now.second * self.fps) + (now.microsecond * self.fps / 1000000))
        sub_frame = 1056964608  # Default value
        frames_packed = struct.pack("!II", frames, sub_frame)

        # Framerate
        frame_rate_packed = struct.pack("!II", self.fps, self._denominator)

        # Blendshapes
        data_packed = struct.pack("!B61f", 61, *self._blend_shapes)

        # Combine all parts
        return (
            version_packed
            + uuid_packed
            + name_length_packed
            + name_packed
            + frames_packed
            + frame_rate_packed
            + data_packed
        )


class LiveLinkDataTrack(FrameProcessor):
    """Processeur qui transmet les visèmes et blendshapes vers Unreal via LiveLink."""

    def __init__(
        self, daily_transport=None, use_udp=True, udp_ip="127.0.0.1", udp_port=11111
    ):
        super().__init__(name="livelink_datatrack")
        self._tx = daily_transport
        self._use_udp = use_udp
        self._face = LiveLinkFace(name="GalaBot")
        self._last_viseme = -1

        # Configuration du dossier de logs
        self._log_dir = Path("livelink_logs")

        # Nettoyer et recréer le dossier de logs
        if self._log_dir.exists():
            shutil.rmtree(self._log_dir)
        self._log_dir.mkdir(exist_ok=True)

        # Configurer le logger de fichier
        self._file_logger = logging.getLogger("livelink")
        self._file_logger.setLevel(logging.INFO)

        # Créer un nouveau fichier de log avec horodatage
        log_file = (
            self._log_dir / f"livelink_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        self._file_logger.addHandler(file_handler)

        # Logger également sur la console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._file_logger.addHandler(console_handler)

        # Socket connection for UDP
        if self._use_udp:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_ip = os.getenv("LIVELINK_IP", udp_ip)
            self._udp_port = int(os.getenv("LIVELINK_PORT", udp_port))
            self._udp_addr = (self._udp_ip, self._udp_port)
            self._file_logger.info(
                f"LiveLinkDataTrack initialisé - UDP {self._udp_ip}:{self._udp_port}"
            )
            try:
                # Envoi d'un paquet vide pour initialiser la connexion
                self._socket.sendto(b"", self._udp_addr)
            except OSError as exc:
                self._file_logger.error(f"Erreur handshake UDP: {exc}")
        else:
            self._file_logger.info("LiveLinkDataTrack initialisé - Daily datatrack")

    async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # D'abord, traiter le frame avec la méthode parente
        await super().process_frame(frame, direction)

        # Traiter les blendshapes complets (prioritaires)
        if isinstance(frame, BlendshapeFrame):
            # Appliquer directement tous les blendshapes
            self._face.set_blendshapes(frame.values)

            # Encoder pour LiveLink
            encoded_data = self._face.encode()

            try:
                if self._use_udp:
                    # Envoi via UDP
                    self._socket.sendto(encoded_data, self._udp_addr)
                elif self._tx:
                    # Envoi via Daily datatrack (format simplifié JSON)
                    msg = json.dumps(
                        {"blendshapes": frame.values, "frame": frame.frame_idx}
                    )
                    await self._tx.send_data(msg.encode(), label="livelink")

                if frame.frame_idx % 10 == 0:  # Log uniquement périodiquement
                    self._file_logger.info(
                        f"LiveLink: envoi frame blendshapes {frame.frame_idx}"
                    )

            except Exception as e:
                self._file_logger.error(f"Erreur LiveLink: {e}")

        # Traiter les visèmes (compatibilité)
        elif isinstance(frame, VisemeFrame):
            viseme_id = frame.id
            time_ms = frame.time_ms

            # Éviter les répétitions inutiles
            if viseme_id == self._last_viseme:
                # Propager le frame sans action
                await self.push_frame(frame, direction)
                return

            self._last_viseme = viseme_id

            # Réinitialiser les blendshapes
            self._face.reset_blendshapes()

            # Configurer les blendshapes en fonction du visème
            # Cette logique dépend de votre modèle - à adapter
            if viseme_id == 0:  # Silence/neutre
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.1)
            elif viseme_id == 1:  # "aa"
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.5)
                self._face.set_blendshape(FaceBlendShape.MouthFunnel, 0.6)
            elif viseme_id == 2:  # "ah"
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.4)
                self._face.set_blendshape(FaceBlendShape.MouthFunnel, 0.4)
            elif viseme_id == 3:  # "oo"
                self._face.set_blendshape(FaceBlendShape.JawOpen, 0.2)
                self._face.set_blendshape(FaceBlendShape.MouthPucker, 0.6)

            # Ajouter un clignotement aléatoire (1 sur 20 frames)
            if viseme_id > 0 and np.random.random() < 0.05:
                self._face.set_blendshape(FaceBlendShape.EyeBlinkLeft, 0.7)
                self._face.set_blendshape(FaceBlendShape.EyeBlinkRight, 0.7)

            # Encoder pour LiveLink
            encoded_data = self._face.encode()

            try:
                if self._use_udp:
                    # Envoi via UDP
                    self._socket.sendto(encoded_data, self._udp_addr)
                elif self._tx:
                    # Envoi via Daily datatrack (format original)
                    msg = json.dumps({"v": viseme_id, "t": time_ms})
                    await self._tx.send_data(msg.encode(), label="livelink")

                self._file_logger.info(
                    f"LiveLink: envoi visème {viseme_id} à t={time_ms}ms"
                )

            except Exception as e:
                self._file_logger.error(f"Erreur LiveLink: {e}")

        # Propager le frame
        await self.push_frame(frame, direction)
