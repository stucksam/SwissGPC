import os
import shutil

import h5py
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment

from src.utils.data_points import DialectDataPoint
from src.utils.logger import get_logger
from src.utils.paths import TTS_PODCASTS_PATH, SCRATCH_PATH, CLUSTER_PROJECTS_TTS

DATASET_NAME = "SDS-200"
SDS200_DATASET_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "sds-200")
SDS200_SCRATCH = os.path.join(SCRATCH_PATH, "sds-200")
SAMPLING_RATE = 16000

CANTON_TO_REGION_ALIGNMENT = {
    "AG": "Zürich",
    "AI": "Ostschweiz",
    "AR": "Ostschweiz",
    "BE": "Bern",
    "BL": "Basel",
    "BS": "Basel",
    "FR": "Bern",
    "GL": "Innerschweiz",
    "GR": "Graubünden",
    "JU": "Bern",
    "LU": "Innerschweiz",
    "NW": "Innerschweiz",
    "OW": "Innerschweiz",
    "SG": "Ostschweiz",
    "SH": "Ostschweiz",
    "SO": "Bern",
    "SZ": "Innerschweiz",
    "TG": "Ostschweiz",
    "UR": "Innerschweiz",
    "VS": "Wallis",
    "ZG": "Innerschweiz",
    "ZH": "Zürich",
}

logger = get_logger(__name__)


def move_sds200_to_h5() -> None:
    logger.info(f"Starting move of {DATASET_NAME} into single h5")
    train_meta_data = load_sds200_train_metadata()
    shutil.copytree(SDS200_DATASET_PATH, SDS200_SCRATCH, dirs_exist_ok=True)

    h5_file_name = f"{DATASET_NAME}.hdf5"
    h5_file_path = os.path.join(TTS_PODCASTS_PATH, h5_file_name)
    meta_data = []

    with h5py.File(h5_file_path, "a") as h5_sds_200:
        for i, sample in enumerate(train_meta_data):
            sample_name_split = sample.sample_name.split("/")
            speaker_folder = sample_name_split[0]
            sample_name = sample_name_split[-1].replace(".mp3", "")
            sample.sample_name = f"{speaker_folder}_{sample_name}" # set correct sample name

            audio_path = os.path.join(SDS200_SCRATCH, speaker_folder, f"{sample_name}.mp3")
            wav_path = audio_path.replace(".mp3", ".wav")

            audio_segment = AudioSegment.from_file(audio_path)
            audio_segment.export(wav_path, format="wav")  # write segment to os for librosa to load in target SR

            speech, _ = librosa.load(wav_path, sr=SAMPLING_RATE)

            h5_entry = h5_sds_200.create_dataset(sample.sample_name, dtype=float, data=speech)
            h5_entry.attrs["dataset_name"] = DATASET_NAME
            h5_entry.attrs["speaker"] = sample.speaker_id
            h5_entry.attrs["duration"] = sample.duration
            h5_entry.attrs["de_text"] = sample.de_text
            h5_entry.attrs["did"] = sample.dialect
            h5_sds_200.flush()

            os.remove(wav_path)  # keep disk space clean
            meta_data.append(sample)

    meta_data_sds200_path = os.path.join(TTS_PODCASTS_PATH, f"{DATASET_NAME}.txt")
    with open(meta_data_sds200_path, "wt", encoding="utf-8") as f:
        f.writelines(sample.to_string() for sample in meta_data)


def load_sds200_train_metadata() -> list[DialectDataPoint]:
    data = []
    train_path = os.path.join(SDS200_DATASET_PATH, "train.tsv")
    df = pd.read_csv(train_path, sep="\t")
    for index, row in df.iterrows():
        canton = row["canton"]
        if canton is None or canton == "" or (isinstance(canton, float) and np.isnan(canton)):
            continue

        data.append(DialectDataPoint(
            dataset_name=DATASET_NAME,
            sample_name=row["clip_path"],  # here clip path, should be fixed afterwards and only sentence_id remain
            duration=row["duration"],
            speaker_id=row["client_id"],
            dialect=CANTON_TO_REGION_ALIGNMENT[canton],
            de_text=row["sentence"],
        ))
    return data
