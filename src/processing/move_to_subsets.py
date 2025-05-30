import os
import random
import shutil
from collections import defaultdict
from multiprocessing import Process

import h5py

from src.transcription.utils import load_meta_data, MISSING_TEXT
from src.utils.data_points import DialectDataPoint
from src.utils.logger import get_logger
from src.utils.paths import SCRATCH_PATH, TTS_PODCASTS_PATH, TTS_TRAINING_SUBSETS_PATH, CLUSTER_PROJECTS_TTS

TARGET_HOURS = 1107  # This is approximately 500GB of audio when sampled at 16kHz
# TARGET_HOURS = 11.07  # This is approximately 5GB of audio when sampled at 16kHz
TARGET_DURATION = TARGET_HOURS * 3600  # convert to seconds
SCRATCH_H5_PATH = os.path.join(SCRATCH_PATH, "transcribed")

logger = get_logger(__name__)


def get_h5_subset_file(idx: int) -> str:
    return os.path.join(SCRATCH_PATH, f"subset_{idx}.hdf5")


def get_podcast_h5_on_scratch(podcast: str) -> str:
    return os.path.join(SCRATCH_H5_PATH, f"{podcast}.hdf5")


def get_subset_metadata(idx: int) -> str:
    return os.path.join(SCRATCH_PATH, f"subset_{idx}.txt")


def write_subset_metadata(samples: list[DialectDataPoint], subset_path: str) -> None:
    with open(subset_path, "wt", encoding="utf-8") as f:
        f.writelines(line.to_string() for line in samples)


def create_h5_subsets(h5_subset_idx: int, samples: list[DialectDataPoint]) -> None:
    logger.info(f"Creating subset {h5_subset_idx} with {len(samples)} samples.")

    # group by podcast to reduce opening and closing podcast h5s due to read operations
    dataset_grouped = defaultdict(list)
    for s in samples:
        dataset_grouped[s.dataset_name].append(s)

    h5_subset_file = get_h5_subset_file(h5_subset_idx)
    meta_data_subset_path = get_subset_metadata(h5_subset_idx)

    if os.path.exists(meta_data_subset_path):
        meta_data_subset, _ = load_meta_data(meta_data_subset_path, load_as_dialect=True)
    else:
        meta_data_subset = []

    with h5py.File(h5_subset_file, "a") as h5_subset:
        for podcast, samples in dataset_grouped.items():
            with h5py.File(get_podcast_h5_on_scratch(podcast), "r") as h5_podcast:
                for i, sample in enumerate(samples):
                    if sample.sample_name in h5_subset:
                        continue

                    h5_content = h5_podcast[sample.sample_name]
                    # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                    new_h5_entry = h5_subset.create_dataset(sample.sample_name, dtype=float, data=h5_content[()])

                    # Copy attributes such as DID, phoneme, mel spec etc.
                    for attr_name, attr_value in h5_content.attrs.items():
                        new_h5_entry.attrs[attr_name] = attr_value

                    h5_subset.flush()
                    meta_data_subset.append(sample)

    write_subset_metadata(meta_data_subset, meta_data_subset_path)
    shutil.copy2(h5_subset_file, TTS_TRAINING_SUBSETS_PATH)
    shutil.copy2(meta_data_subset_path, TTS_TRAINING_SUBSETS_PATH)

    logger.info(f"Finished creation of subset {h5_subset_idx}.")


def move_podcasts_to_subset() -> None:
    shutil.copytree(TTS_PODCASTS_PATH, SCRATCH_H5_PATH, dirs_exist_ok=True)
    # shutil.copytree(os.path.join(CLUSTER_PROJECTS_TTS, "test_h5_dir"), SCRATCH_H5_PATH, dirs_exist_ok=True)

    # Load all metadata and shuffle samples
    metadata_files = [file for file in os.listdir(SCRATCH_H5_PATH) if file.endswith(".txt")]

    all_samples = []
    for metadata_file in metadata_files:
        podcast_samples, _ = load_meta_data(os.path.join(SCRATCH_H5_PATH, metadata_file))
        for sample in podcast_samples:
            sample.dataset_name = metadata_file.replace(".txt", "")

        podcast_samples = [sample.convert_to_dialect_datapoint()
                           for sample in podcast_samples
                           if sample.de_text != MISSING_TEXT
                           and sample.dialect and sample.dialect != "English"
                           and sample.dataset_name]

        assert len(podcast_samples) > 0, (f"Filtering lead to no actual samples being loaded for move to subset h5s"
                                          f"in podcast {metadata_file}.")
        all_samples.extend(podcast_samples)

    logger.info(f"Collected {len(all_samples)} samples from {len(metadata_files)} podcasts.")

    random.seed(18670209)  # Sōseki!
    random.shuffle(all_samples)

    # Create groups
    grouped_samples = []
    current_group = []
    current_duration = 0.0

    for sample in all_samples:
        current_group.append(sample)
        current_duration += sample.duration

        if current_duration >= TARGET_DURATION:
            grouped_samples.append(current_group)
            current_group = []
            current_duration = 0.0

    # Handle the last group
    if current_group:
        grouped_samples.append(current_group)

    os.makedirs(TTS_TRAINING_SUBSETS_PATH, exist_ok=True)

    processes = [
        Process(target=create_h5_subsets, args=(i, group))
        for i, group in enumerate(grouped_samples) if len(group) > 1000  # if contains entries then make process
        # for i, group in enumerate(grouped_samples)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
