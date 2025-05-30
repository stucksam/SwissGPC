import json
import os

import h5py
import pandas as pd

from src.processing.utils import SNF_DATASET_PATH
from src.transcription.utils import DIALECT_TO_TAG
from src.utils.data_points import DialectDataPoint
from src.utils.logger import get_logger
from src.utils.paths import TTS_PODCASTS_PATH

DATASET_NAME = "STT4SG-350"
logger = get_logger(__name__)


def move_stt4sg_to_h5():
    logger.info(f"Starting move of {DATASET_NAME} into single h5")

    with open(os.path.join(SNF_DATASET_PATH, "speaker_to_dialect.json"), "rt", encoding="utf-8") as f:
        speaker_to_dialect = json.loads(f.read())

    dialects = {key: [] for key in DIALECT_TO_TAG.keys()}

    # Group metadata by dialect
    for speaker, dialect in speaker_to_dialect.items():
        dialects[dialect].append(speaker)

    h5_file_name = f"{DATASET_NAME}.hdf5"
    h5_file_path = os.path.join(TTS_PODCASTS_PATH, h5_file_name)
    meta_data = []

    with h5py.File(h5_file_path, "a") as h5_stt4sg:
        for dialect, speakers in dialects.items():
            for speaker in speakers:
                speaker_path = f"{SNF_DATASET_PATH}/speakers/{speaker}"
                meta_data_speaker = create_datapoints_for_stt4sg_corpus_speaker(dialect, speaker, speaker_path, True)
                with h5py.File(f"{speaker_path}/audio.h5", "r") as h5_read:
                    for entry in meta_data_speaker:
                        # I want uniformity in hdf5 keys of type SAMPLE_CUTID with only one underscore or just SAMPLE
                        new_sample_name = entry.sample_name.split("-")[-1]
                        if new_sample_name in h5_stt4sg:
                            continue

                        h5_content = h5_read[entry.sample_name]
                        # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                        new_h5_entry = h5_stt4sg.create_dataset(new_sample_name, dtype=float, data=h5_content[()])

                        # Create essential attributes
                        new_h5_entry.attrs["dataset_name"] = entry.dataset_name
                        new_h5_entry.attrs["speaker"] = entry.speaker_id
                        new_h5_entry.attrs["de_text"] = entry.de_text
                        new_h5_entry.attrs["did"] = dialect
                        h5_stt4sg.flush()

                        entry.sample_name = new_sample_name
                        meta_data.append(entry)

    meta_data_stt4sg_path = os.path.join(TTS_PODCASTS_PATH, f"{DATASET_NAME}.txt")
    with open(meta_data_stt4sg_path, "wt", encoding="utf-8") as f:
        f.writelines(sample.to_string() for sample in meta_data)

    logger.info(f"Finished move for {DATASET_NAME} to hdf5.")


def _load_snf_tsv_for_duration() -> dict:
    train = os.path.join(SNF_DATASET_PATH, "train_all.tsv")
    test = os.path.join(SNF_DATASET_PATH, "test.tsv")
    valid = os.path.join(SNF_DATASET_PATH, "valid.tsv")
    sample_to_duration = {}

    for file in [train, test, valid]:
        df = pd.read_csv(file, sep="\t")
        for index, row in df.iterrows():
            # path thingy is something custom because I just copy ready made h5s, check your env and replace as needed
            sample_to_duration[row["path"].replace("/", "-").replace(".flac", "")] = round(float(row["duration"]), 4)
    return sample_to_duration


def create_datapoints_for_stt4sg_corpus_speaker(dialect: str, speaker: str, speaker_path: str,
                                                parse_duration: bool = False) -> list[DialectDataPoint]:
    """
    As STT4SG corpus was prepared to be used with a speaker based folder structure its required to parse through
    each speaker and load their samples into the hdf5
    :param dialect:
    :param speaker:
    :param speaker_path:
    :param parse_duration:
    :return:
    """
    data = []
    if parse_duration:
        sample_to_duration = _load_snf_tsv_for_duration()
    with open(f"{speaker_path}/metadata.txt", "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.split("|")
            sample_name = split_line[0]
            data.append(DialectDataPoint(
                dataset_name=DATASET_NAME,
                sample_name=sample_name,
                duration=sample_to_duration[sample_name] if parse_duration else -1.0,
                speaker_id=speaker,
                dialect=dialect,
                de_text=split_line[1],
            ))
    return data
