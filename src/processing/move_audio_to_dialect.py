import json
import os
from multiprocessing import Process

import h5py
import pandas as pd

from src.processing.utils import SWISSDIAL_CANTON_TO_DIALECT, SWISSDIAL_DATASET_PATH, SNF_DATASET_PATH
from src.transcription.utils import DIALECT_DATA_PATH, load_meta_data, get_h5_file, get_metadata_path, DIALECT_TO_TAG
from src.utils.data_points import DialectDataPoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_dialect_meta_data_path(dialect: str) -> str:
    return os.path.join(DIALECT_DATA_PATH, f"{dialect}.txt")


def write_dialect_meta_data(dialect: str, dialect_content: list[DialectDataPoint]) -> None:
    with open(get_dialect_meta_data_path(dialect), "wt", encoding="utf-8") as f:
        f.writelines(line.to_string() for line in dialect_content)


def get_dialect_h5_path(dialect: str) -> str:
    return os.path.join(DIALECT_DATA_PATH, f"{dialect}.hdf5")


def get_dialect_files(dialect) -> tuple[list, str]:
    meta_data_dialect_path = get_dialect_meta_data_path(dialect)
    h5_file_dialect = get_dialect_h5_path(dialect)

    if os.path.exists(meta_data_dialect_path):
        meta_data_dialect, _ = load_meta_data(meta_data_dialect_path)
    else:
        meta_data_dialect = []

    return meta_data_dialect, h5_file_dialect


def start_podcast_move(podcast: str, dialect: str, samples: list) -> None:
    logger.info(f"Performing move for dialect '{dialect}'.")
    h5_file_podcast = get_h5_file(podcast)
    with h5py.File(h5_file_podcast, "r") as h5_podcast:
        meta_data_dialect, h5_file_dialect = get_dialect_files(dialect)

        with h5py.File(h5_file_dialect, "a") as h5_dialect:
            for entry in samples:
                if entry.sample_name in h5_dialect:
                    continue

                h5_content = h5_podcast[entry.sample_name]
                # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                new_h5_entry = h5_dialect.create_dataset(entry.sample_name, dtype=float, data=h5_content[()])

                # Copy attributes such as DID, phoneme, mel spec etc.
                for attr_name, attr_value in h5_content.attrs.items():
                    new_h5_entry.attrs[attr_name] = attr_value

                h5_dialect.flush()
                entry.dataset_name = podcast
                meta_data_dialect.append(entry.convert_to_dialect_datapoint())

        write_dialect_meta_data(dialect, meta_data_dialect)

    logger.info(f"Finished move for dialect '{dialect}'.")


def move_podcast_to_dialect(podcast: str) -> None:
    """
    Runs move of dialect data in parallel to reduce time
    :param podcast:
    :return:
    """
    logger.info(f"Starting concurrent move of podcast '{podcast}' to dialect hdf5.")

    # Load metadata and initialize dialects
    meta_data, _ = load_meta_data(get_metadata_path(podcast))
    dialects = {key: [] for key in DIALECT_TO_TAG.keys()}
    os.makedirs(DIALECT_DATA_PATH, exist_ok=True)

    # Group metadata by dialect
    for entry in meta_data:
        dialects[entry.dialect].append(entry)

    processes = [
        Process(target=start_podcast_move, args=(podcast, dialect, samples))
        for dialect, samples in dialects.items() if samples  # if contains entries then make process
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


def _load_snf_tsv_for_duration() -> dict:
    path = "src/config"
    train = f"{path}/train_all.tsv"
    test = f"{path}/test.tsv"
    valid = f"{path}/valid.tsv"
    sample_to_duration = {}

    for file in [train, test, valid]:
        df = pd.read_csv(file, sep="\t")
        for index, row in df.iterrows():
            # path thingy is something custom because I just copy ready made h5s, check your env and replace as needed
            sample_to_duration[row["path"].replace("/", "-").replace(".flac", "")] = round(float(row["duration"]), 4)
    return sample_to_duration


def create_datapoints_for_stt4sg_corpus_speaker(speaker: str, speaker_path: str, parse_duration: bool = False) -> list[
    DialectDataPoint]:
    """
    As SNF / STT4SG corpus was prepared to be used with a speaker based folder structure its required to parse through
    each speaker and load their samples into the hdf5
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
                dataset_name="SNF",
                sample_name=sample_name,
                duration=sample_to_duration[sample_name] if parse_duration else -1.0,
                speaker_id=speaker,
                de_text=split_line[1],
            ))
    return data


def load_stt4sg_speakers_to_dialect(dialect: str, speakers: list) -> None:
    logger.info(f"Starting move for dialect '{dialect}' for SNF.")
    meta_data_dialect, h5_file_dialect = get_dialect_files(dialect)

    with h5py.File(h5_file_dialect, "a") as h5_dialect:
        for speaker in speakers:
            speaker_path = f"{SNF_DATASET_PATH}/speakers/{speaker}"
            meta_data_speaker = create_datapoints_for_stt4sg_corpus_speaker(speaker, speaker_path, True)
            with h5py.File(f"{speaker_path}/audio.h5", "r") as h5_read:
                for entry in meta_data_speaker:
                    # I want uniformity in hdf5 keys of type SAMPLE_CUTID with only one underscore or just SAMPLE
                    new_sample_name = entry.sample_name.split("-")[-1]
                    if new_sample_name in h5_dialect:
                        continue

                    h5_content = h5_read[entry.sample_name]
                    # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                    new_h5_entry = h5_dialect.create_dataset(new_sample_name, dtype=float, data=h5_content[()])

                    # Create essential attributes
                    new_h5_entry.attrs["dataset_name"] = entry.dataset_name
                    new_h5_entry.attrs["speaker"] = entry.speaker_id
                    new_h5_entry.attrs["de_text"] = entry.de_text
                    new_h5_entry.attrs["did"] = dialect
                    h5_dialect.flush()

                    entry.sample_name = new_sample_name
                    meta_data_dialect.append(entry)

        write_dialect_meta_data(dialect, meta_data_dialect)
        logger.info(f"Finished move for dialect '{dialect}'.")


def move_stt4sg_corpus_to_dialect() -> None:
    with open(os.path.join(SNF_DATASET_PATH, "speaker_to_dialect.json"), "rt", encoding="utf-8") as f:
        speaker_to_dialect = json.loads(f.read())

    dialects = {key: [] for key in DIALECT_TO_TAG.keys()}
    os.makedirs(DIALECT_DATA_PATH, exist_ok=True)

    # Group metadata by dialect
    for speaker, dialect in speaker_to_dialect.items():
        dialects[dialect].append(speaker)

    processes = [
        Process(target=load_stt4sg_speakers_to_dialect, args=(dialect, samples))
        for dialect, samples in dialects.items() if samples  # if contains entries then make process
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


def create_datapoint_for_canton(canton: str) -> list[DialectDataPoint]:
    """
    As there is only one speaker per canton the speaker receives a unique name consisting of SPEAKER_ch_canton.
    :param canton:
    :return:
    """
    data = []
    canton_path = f"{SWISSDIAL_DATASET_PATH}/{canton}"
    with open(f"{canton_path}/metadata.txt", "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.split("|")
            data.append(DialectDataPoint(
                dataset_name="SwissDial",
                sample_name=split_line[0],
                duration=-1.0,
                speaker_id=f"SPEAKER_ch_{canton}",
                de_text=split_line[1],
            ))
    return data


def move_swissdial_canton_to_dialect(canton: str) -> None:
    dialect = SWISSDIAL_CANTON_TO_DIALECT[canton]
    logger.info(f"Starting move for dialect '{dialect}' for SwissDial.")

    meta_data = create_datapoint_for_canton(canton)
    canton_path = f"{SWISSDIAL_DATASET_PATH}/{canton}"

    with h5py.File(f"{canton_path}/audio.h5", "r") as h5_read:
        meta_data_dialect, h5_file_dialect = get_dialect_files(dialect)

        with h5py.File(h5_file_dialect, "a") as h5_dialect:
            for entry in meta_data:

                # I want uniformity in hdf5 keys of type SAMPLE_CUTID with only one underscore
                new_sample_name = entry.sample_name.replace(f"ch_{canton}", f"ch-{canton}")

                if new_sample_name in h5_dialect:
                    continue

                h5_content = h5_read[entry.sample_name]
                # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                new_h5_entry = h5_dialect.create_dataset(new_sample_name, dtype=float, data=h5_content[()])

                # Create essential attributes
                new_h5_entry.attrs["dataset_name"] = entry.dataset_name
                new_h5_entry.attrs["speaker"] = entry.speaker_id
                new_h5_entry.attrs["de_text"] = entry.de_text
                new_h5_entry.attrs["did"] = dialect
                h5_dialect.flush()

                entry.sample_name = new_sample_name
                meta_data_dialect.append(entry)

        write_dialect_meta_data(dialect, meta_data_dialect)

    logger.info(f"Finished move for dialect '{dialect}'.")


def move_swissdial_to_dialect() -> None:
    aargau = "ag"
    processes = [
        Process(target=move_swissdial_canton_to_dialect, args=(canton,))
        for canton, dialect in SWISSDIAL_CANTON_TO_DIALECT.items() if canton != aargau
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    move_swissdial_canton_to_dialect(aargau)  # wait for Zurich to be done so no issues with concurrency occur
