import json
import os

import pandas as pd

from src.transcription.utils import load_meta_data
from src.utils.logger import get_logger
from src.utils.paths import PODCAST_AUDIO_FOLDER, TTS_RAW_AUDIO_PATH

logger = get_logger(__name__)


def calc_podcast_duration():
    files = [file for file in os.listdir(PODCAST_AUDIO_FOLDER) if file.endswith(".txt")]

    duration = {
        "podcast": [],
        "num_samples": [],
        "duration": []
    }
    for podcast in files:
        meta_data, num_samples = load_meta_data(os.path.join(PODCAST_AUDIO_FOLDER, podcast), load_as_dialect=False)
        dur = 0.0
        for sample in meta_data:
            dur += sample.duration
        dur = round(dur / 3600, 4)
        duration["podcast"].append(podcast)
        duration["num_samples"].append(num_samples)
        duration["duration"].append(dur)
        logger.info(f"{podcast}: {dur}")

    df = pd.DataFrame(duration)
    df.to_csv("duration.csv", index=False, sep=";", encoding="utf-8")


def count_raw_audio():
    pods = [os.path.join(TTS_RAW_AUDIO_PATH, d) for d in os.listdir(TTS_RAW_AUDIO_PATH) if os.path.isdir(os.path.join(TTS_RAW_AUDIO_PATH, d))]
    logger.info(len(pods))
    duration = {
        "podcast": [],
        "num_episodes": [],
        "duration": []
    }
    for pod in pods:
        podcast_name = pod.split("/")[-1]
        logger.info(f"Processing {podcast_name}")
        dur = 0.0
        jsons = [os.path.join(pod, f) for f in os.listdir(pod)
                 if os.path.isfile(os.path.join(pod, f))
                 and f.endswith(".json")]

        for json_file in jsons:
            with open(json_file, "r", encoding='utf8') as f:
                segments = json.load(f)
            end = float(segments[-1]["end"])
            dur += end

        dur = round(dur / 3600, 4)
        duration["podcast"].append(pod)
        duration["num_episodes"].append(len(jsons))
        duration["duration"].append(dur)

        logger.info(f"{podcast_name}: {dur}")

    df = pd.DataFrame(duration)
    df.to_csv("duration_complete.csv", index=False, sep=";", encoding="utf-8")


def distribution_of_dialects():
    files = [file for file in os.listdir(PODCAST_AUDIO_FOLDER) if file.endswith(".txt")]
    duration = {
        "podcast": [],
        "duration": [],
        "dur_zürich": [],
        "dur_innerschweiz": [],
        "dur_ostschweiz": [],
        "dur_bern": [],
        "dur_basel": [],
        "dur_wallis": [],
        "dur_graubünden": [],
        "dur_deutschland": [],
        "dur_english": []
    }

    for podcast in files:
        logger.info(f"Processing {podcast}.")
        meta_data, _ = load_meta_data(os.path.join(PODCAST_AUDIO_FOLDER, podcast), load_as_dialect=False)
        dur = 0.0
        dialects = {
            "zürich": 0.0,
            "innerschweiz": 0.0,
            "ostschweiz": 0.0,
            "bern": 0.0,
            "basel": 0.0,
            "wallis": 0.0,
            "graubünden": 0.0,
            "deutschland": 0.0,
            "english": 0.0
        }
        for sample in meta_data:
            dur += sample.duration
            dialects[sample.dialect.lower()] += sample.duration

        dur = round(dur / 3600, 4)
        dialects = {k: round(v / 3600, 4) for k, v in dialects.items()}
        duration["podcast"].append(podcast)
        duration["duration"].append(dur)
        for dialect, dialect_duration in dialects.items():
            duration[f"dur_{dialect.lower()}"].append(dialect_duration)

    df = pd.DataFrame(duration)
    df.to_csv("dialect_durations.csv", index=False, sep=";", encoding="utf-8")
