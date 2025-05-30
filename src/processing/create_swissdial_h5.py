import os

import h5py

from src.processing.utils import SWISSDIAL_DATASET_PATH, SWISSDIAL_CANTON_TO_DIALECT
from src.utils.data_points import DialectDataPoint
from src.utils.logger import get_logger
from src.utils.paths import TTS_PODCASTS_PATH

DATASET_NAME = "SwissDial"
logger = get_logger(__name__)


def create_datapoint_for_canton(canton: str) -> list[DialectDataPoint]:
    """
    As there is only one speaker per canton the speaker receives a unique name consisting of SPEAKER_ch_canton.
    :param canton:
    :return:
    """
    data = []
    canton_path = os.path.join(SWISSDIAL_DATASET_PATH, canton)
    with open(f"{canton_path}/metadata.txt", "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.split("|")
            data.append(DialectDataPoint(
                dataset_name=DATASET_NAME,
                sample_name=split_line[0],
                duration=0.0,
                speaker_id=f"SPEAKER_ch_{canton}",
                dialect=SWISSDIAL_CANTON_TO_DIALECT[canton],
                de_text=split_line[1],
            ))
    return data


def move_swissdial_to_h5() -> None:
    logger.info(f"Starting move of {DATASET_NAME} into single h5")

    h5_file_name = f"{DATASET_NAME}.hdf5"
    h5_file_path = os.path.join(TTS_PODCASTS_PATH, h5_file_name)
    swiss_dial_meta_data  = []

    with h5py.File(h5_file_path, "a") as h5_swiss_dial:
        for canton in SWISSDIAL_CANTON_TO_DIALECT.keys():

            dialect = SWISSDIAL_CANTON_TO_DIALECT[canton]
            meta_data_canton = create_datapoint_for_canton(canton)
            canton_audio_path = os.path.join(SWISSDIAL_DATASET_PATH, canton, "audio.h5")

            with h5py.File(canton_audio_path, "r") as h5_read:

                for entry in meta_data_canton:
                    # I want uniformity in hdf5 keys of type SAMPLE_CUTID with only one underscore
                    new_sample_name = entry.sample_name.replace(f"ch_{canton}", f"ch-{canton}")
                    if new_sample_name in h5_swiss_dial:
                        continue

                    h5_content = h5_read[entry.sample_name]
                    # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                    new_h5_entry = h5_swiss_dial.create_dataset(new_sample_name, dtype=float, data=h5_content[()])

                    # Create essential attributes
                    new_h5_entry.attrs["dataset_name"] = entry.dataset_name
                    new_h5_entry.attrs["speaker"] = entry.speaker_id
                    new_h5_entry.attrs["de_text"] = entry.de_text
                    new_h5_entry.attrs["did"] = dialect
                    h5_swiss_dial.flush()

                    entry.sample_name = new_sample_name
                    swiss_dial_meta_data.append(entry)

    meta_data_swiss_dial_path = os.path.join(TTS_PODCASTS_PATH, f"{DATASET_NAME}.txt")
    with open(meta_data_swiss_dial_path, "wt", encoding="utf-8") as f:
        f.writelines(sample.to_string() for sample in swiss_dial_meta_data)

    logger.info(f"Finished move for {DATASET_NAME} to hdf5.")
