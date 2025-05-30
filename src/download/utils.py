import os

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import PODCAST_METADATA_FOLDER, PODCAST_AUDIO_FOLDER

logger = get_logger(__name__)


def save_podcast_metadata_to_csv(podcast: str, episodes: list) -> None:
    episodes = pd.DataFrame(episodes)
    episodes.to_csv(os.path.join(PODCAST_METADATA_FOLDER, podcast + ".csv"), index=False, sep=";", encoding="utf-8")


def load_podcast_metadata_from_csv(podcast: str) -> pd.DataFrame:
    podcast = podcast if podcast.endswith(".csv") else podcast + ".csv"
    return pd.read_csv(os.path.join(PODCAST_METADATA_FOLDER, podcast), encoding="utf-8", sep=";")


def create_audio_folder_if_not_exists(podcast):
    os.makedirs(PODCAST_AUDIO_FOLDER, exist_ok=True)
    podcast_path = os.path.join(PODCAST_AUDIO_FOLDER, podcast)
    os.makedirs(podcast_path, exist_ok=True)


def get_podcast_path(podcast: str) -> str:
    return os.path.join(PODCAST_AUDIO_FOLDER, podcast)


def get_downloaded_metadata() -> list[str]:
    os.makedirs(PODCAST_METADATA_FOLDER, exist_ok=True)
    return [f for f in os.listdir(PODCAST_METADATA_FOLDER) if os.path.isfile(os.path.join(PODCAST_METADATA_FOLDER, f))]


def get_duration_podcasts():
    total_duration = 0.0
    skipped = 0

    for podcast in os.listdir(PODCAST_METADATA_FOLDER):
        df = load_podcast_metadata_from_csv(podcast)
        episode_durations = []

        for _, metadata in df.iterrows():
            duration = metadata.get("duration_ms") or metadata.get("duration_s")

            if duration == "NO_DURATION":
                skipped += 1
                continue

            episode_durations.append(int(duration))

        podcast_duration = sum(episode_durations)
        logger.info(f"{podcast}: {podcast_duration}s; {round(podcast_duration / 3600, 4)}h.")

        total_duration += podcast_duration

    logger.info(f"Total duration: {total_duration}s; {round(total_duration / 3600, 4)}h.")
    logger.info(f"Skipped {skipped} episodes.")
