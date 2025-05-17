import asyncio
from src.gala.livelink import LiveLinkDataTrack, BlendshapeFrame


async def main():
    track = LiveLinkDataTrack(use_udp=True)
    # Exemple de frame avec valeurs neutres
    frame = BlendshapeFrame([0.0] * 61, frame_idx=0)
    await track.process_frame(frame)


if __name__ == "__main__":
    asyncio.run(main())
