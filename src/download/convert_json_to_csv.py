import json
import os

import pandas as pd

from src.download.utils import PODCAST_METADATA_FOLDER


def convert_all_json():
    files = [file for file in os.listdir(PODCAST_METADATA_FOLDER) if file.endswith(".json")]
    for file in files:
        podcast_name = file.split(".json")[0]
        convert_json_metadata_to_csv(podcast_name)


def convert_json_metadata_to_csv(podcast: str) -> None:
    with open(os.path.join(PODCAST_METADATA_FOLDER, podcast + ".json"), "r", encoding="utf-8") as f:
        content = json.loads(f.read())

    df_content = []
    for e_id, metadata in content.items():
        df_content.append({
            "id": e_id,
            "title": metadata["title"],
            "description": metadata["description"].replace("\r\n", " ").replace("\n", " ").replace("_ ", ""),
            "date_published": metadata["date_published"],
            "duration_s": int(metadata["duration_ms"]) / 1000,
            # "download_available": metadata["download_available"],
            "url": metadata["url"]
        })
    df = pd.DataFrame(df_content)
    df.to_csv(os.path.join(PODCAST_METADATA_FOLDER, podcast + ".csv"), index=False, sep=";", encoding="utf-8")
