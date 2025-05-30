import os

import torch

from src.utils.data_points import DatasetDataPoint, DialectDataPoint
from src.utils.logger import get_logger
from src.utils.paths import PODCAST_AUDIO_FOLDER

DIALECT_DATA_PATH = os.path.join(PODCAST_AUDIO_FOLDER, "dialects")

META_WRITE_ITERATIONS = 25
MISSING_TEXT = "NO_TEXT"
DIALECT_TO_TAG = {
    "Zürich": "ch_zh",
    "Innerschweiz": "ch_lu",
    "Wallis": "ch_vs",
    "Graubünden": "ch_gr",
    "Ostschweiz": "ch_sg",
    "Basel": "ch_bs",
    "Bern": "ch_be",
    "Deutschland": "de"
}

logger = get_logger(__name__)


def get_metadata_path(podcast: str) -> str:
    """
    Loads text file to Python object for metadata handling.
    :param podcast: Name of podcast or dialect that needs to be loaded
    :return:
    """
    return os.path.join(PODCAST_AUDIO_FOLDER, f"{podcast}.txt")


def load_meta_data(meta_data_path: str, load_as_dialect: bool = False) -> tuple[
    list[DatasetDataPoint | DialectDataPoint], int]:
    """
    Loads podcast metadata generated on hdf5 file creation or enriched during subsequent processes. It's expected
    that metadata files do no contain mixed Dialect separated (read: reduced) metadata and Dataset separated
    (read: detailed) metadata.
    :param meta_data_path: Path to metadata file which should be parsed
    :return:
    """
    sample_list = []
    with open(meta_data_path, "rt", encoding="utf-8") as meta_file:
        for line in meta_file:
            split_line = line.replace('\n', '').split('\t')
            if DIALECT_DATA_PATH in meta_data_path or load_as_dialect:
                sample_list.append(DialectDataPoint.load_single_datapoint(split_line))
            else:
                sample_list.append(DatasetDataPoint.load_single_datapoint(split_line))

    length_samples = len(sample_list)
    # random.shuffle(sample_list)
    logger.info(f"NO. OF SAMPLES: {length_samples}")
    return sample_list, length_samples


def write_meta_data(podcast: str, meta_data) -> None:
    with open(get_metadata_path(podcast), "wt", encoding="utf-8") as f:
        for line in meta_data:
            f.write(line.to_string())


def get_h5_file(podcast: str) -> str:
    return os.path.join(PODCAST_AUDIO_FOLDER, f"{podcast}.hdf5")


def setup_gpu_device() -> tuple:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Training / Inference device is: {device}")
    return device, torch_dtype
