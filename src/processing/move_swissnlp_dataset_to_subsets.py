import os
import random
from collections import defaultdict

import h5py

from src.transcription.utils import load_meta_data, MISSING_TEXT
from src.utils.logger import get_logger
from src.utils.paths import TTS_PODCASTS_PATH, TTS_TRAINING_SUBSETS_PATH

DATASETS = ["SwissDial", "SDS-200", "STT4SG-350"]
NUM_SUBSETS = 8

logger = get_logger(__name__)


def move_to_h5(grouped_samples: list) -> None:
    for i, group in enumerate(grouped_samples):
        dataset_grouped = defaultdict(list)
        for s in group:
            dataset_grouped[s.dataset_name].append(s)

        h5_subset_file = os.path.join(TTS_TRAINING_SUBSETS_PATH, f"subset_{i}.hdf5")
        meta_data_subset_path = os.path.join(TTS_TRAINING_SUBSETS_PATH, f"subset_{i}.txt")

        if os.path.exists(meta_data_subset_path):
            meta_data_subset, _ = load_meta_data(meta_data_subset_path, load_as_dialect=True)
        else:
            meta_data_subset = []

        if not meta_data_subset:
            logger.error("Tried to create new h5 group")
            return

        with h5py.File(h5_subset_file, "a") as h5_subset:
            for podcast, samples in dataset_grouped.items():
                swissnlp_dataset_h5_path = os.path.join(TTS_PODCASTS_PATH, f"{podcast}.hdf5")

                with h5py.File(swissnlp_dataset_h5_path, "r") as h5_swiss_nlp:
                    for sample in samples:
                        if sample.sample_name in h5_subset:
                            logger.warning(f"Sample {sample.sample_name} already exists in subset file. "
                                           f"Skipping or overwriting.")
                            # Optionally: del h5_subset[sample.sample_name]

                        h5_content = h5_swiss_nlp[sample.sample_name]
                        # specifically using [()] instead of [:] to reduce operation time as we are not slicing
                        new_h5_entry = h5_subset.create_dataset(sample.sample_name, dtype=float, data=h5_content[()])

                        # Copy attributes such as DID, phoneme, mel spec etc.
                        for attr_name, attr_value in h5_content.attrs.items():
                            new_h5_entry.attrs[attr_name] = attr_value

                        h5_subset.flush()
                        meta_data_subset.append(sample)

        with open(meta_data_subset_path, "wt", encoding="utf-8") as f:
            f.writelines(line.to_string() for line in meta_data_subset)


def add_swissdial_dataset(grouped_samples: list) -> list:
    podcast_samples, _ = load_meta_data(os.path.join(TTS_PODCASTS_PATH, f"SwissDial.txt"), load_as_dialect=True)
    random.shuffle(podcast_samples)

    n = len(podcast_samples)
    number_of_samples_per_subset = n // NUM_SUBSETS + 10 # I dont want to loose any data

    for i, group in enumerate(grouped_samples):
        logger.info(f"Group {i} at start: {len(group)}")
        start = i * number_of_samples_per_subset
        if i == NUM_SUBSETS - 1:
            # Last group gets everything that's left
            group.extend(podcast_samples[start:])
        else:
            end = start + number_of_samples_per_subset
            group.extend(podcast_samples[start:end])
            logger.info(f"Added {end-start} samples to group {i}")
        logger.info(f"Group {i} at end: {len(group)}")

    return grouped_samples


def move_swissnlp_datasets():
    all_samples = []

    for dataset in ["SDS-200", "STT4SG-350"]:
        podcast_samples, _ = load_meta_data(os.path.join(TTS_PODCASTS_PATH, f"{dataset}.txt"), load_as_dialect=True)

        for sample in podcast_samples:
            sample.dataset_name = dataset

        podcast_samples = [sample
                           for sample in podcast_samples
                           if sample.de_text != MISSING_TEXT
                           and sample.dialect and sample.dialect != "English"
                           and sample.dataset_name]

        assert len(podcast_samples) > 0, (f"Filtering lead to no actual samples being loaded for move to subset h5s"
                                          f"in podcast {dataset}.")

        all_samples.extend(podcast_samples)

    total_duration = round(sum([float(sample.duration) for sample in all_samples]) / 3600, 4)
    logger.info(
        f"Collected {len(all_samples)} samples from {len(DATASETS)} podcasts with total duration: {total_duration}h.")

    target_duration = (round(total_duration / NUM_SUBSETS, 4) + 0.5) * 3600
    logger.info(f"Target duration is {target_duration}")
    random.seed(18670209)  # Sōseki!
    random.shuffle(all_samples)

    grouped_samples = []
    current_group = []
    current_duration = 0.0

    for sample in all_samples:
        current_group.append(sample)
        current_duration += float(sample.duration)

        if current_duration >= target_duration:
            grouped_samples.append(current_group)
            current_group = []
            current_duration = 0.0

    if current_group:
        grouped_samples.append(current_group)
    grouped_samples = add_swissdial_dataset(grouped_samples)

    move_to_h5(grouped_samples)
