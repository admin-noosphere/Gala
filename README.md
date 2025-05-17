# Gala

Agent vocal utilisant Pipecat et NeuroSync.

## Installation

pip install -r requirements.txt
pip install -e ./vendor/pipecat[webrtc,daily,deepgram,elevenlabs,openai,silero]
pip install -r requirements-dev.txt

# Service OpenAI TTS pour Gala

Ce module fournit une intégration du service de synthèse vocale (TTS) d'OpenAI avec Pipecat pour le projet Gala.

## Fonctionnalités

- Intégration complète avec le pipeline Pipecat
- Support des modèles OpenAI TTS (gpt-4o-mini-tts et tts-1)
- Choix de différentes voix (alloy, echo, fable, onyx, nova, shimmer, etc.)
- Instructions de style pour personnaliser la voix
- Streaming audio pour une réponse rapide
- Envoi automatique de l'audio vers NeuroSync pour la synchronisation labiale

## Configuration

1. Assurez-vous d'avoir une clé API OpenAI valide dans votre fichier `.env` :
   ```
   OPENAI_API_KEY=sk-...
   ```

2. Pour utiliser OpenAI TTS au lieu d'ElevenLabs, configurez la variable d'environnement :
   ```
   TTS_SERVICE=openai
   ```



## Scripts d'exemple


## Monitoring externe

Pour observer l'activité de Gala il est possible de développer un petit tableau de bord Web.
Une approche simple consiste à créer une application [Next.js](https://nextjs.org/) utilisant la librairie de composants [shadcn/ui](https://ui.shadcn.com/).

Cette page pourrait afficher en temps réel :

- L'état de connexion à la room Daily
- Les transcriptions reçues par Deepgram
- Les réponses générées par l'LLM
- Les statistiques d'usage (tokens et durée de parole)

Le serveur Python peut exposer ces informations via WebSocket ou SSE afin d'alimenter la page Next.js.