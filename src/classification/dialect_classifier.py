import os
from collections import Counter, defaultdict

from joblib import load

from src.transcription.utils import load_meta_data, get_metadata_path, write_meta_data
from src.utils.logger import get_logger
from src.utils.paths import MODEL_PATH

BATCH_SIZE = 32
# MODEL_PATH_DID = os.path.join(MODEL_PATH, "did", "text_clf_3_ch_de.joblib")
MODEL_PATH_DID = os.path.join(MODEL_PATH, "did", "text_clf_de_eng_ch.joblib")

PHON_DID_CLS = {0: "Zürich", 1: "Innerschweiz", 2: "Wallis", 3: "Graubünden", 4: "Ostschweiz", 5: "Basel", 6: "Bern",
                7: "Deutschland", 8: "English"}

logger = get_logger(__name__)


class MergingHelper:

    def __init__(self, sample):
        self.duration: float = sample.duration
        self.samples: list = [sample]


def merge_phoneme_of_speaker_samples(combinations: list[MergingHelper]):
    texts = []
    for entries in combinations:
        texts.append(" ".join([cut.phoneme.replace(' ', '') for cut in entries.samples]))
    return texts


def assign_samples_to_speaker(meta_data: list, max_length: float = 30.0) -> dict:
    """
    merge together samples max_length seconds of speakers.
    :return:
    """
    speaker_to_episodes = defaultdict(lambda: defaultdict(list))
    for sample in meta_data:
        episode = speaker_to_episodes[sample.orig_episode_name][sample.speaker_id]

        for group in episode:
            if group.duration + sample.duration <= max_length:
                group.duration += sample.duration
                group.samples.append(sample)
                break
        else:
            episode.append(MergingHelper(sample))

    return speaker_to_episodes


def dialect_identification_naive_bayes_majority_voting(podcast: str) -> None:
    logger.info("Run Dialect Identification based on phonemes with Majority Voting of 100s samples")
    meta_data, _ = load_meta_data(get_metadata_path(podcast))
    speaker_merged_phoneme = assign_samples_to_speaker(meta_data, max_length=100.0)

    text_clf = load(MODEL_PATH_DID)
    text_clf["clf"].set_params(n_jobs=BATCH_SIZE)

    # since Python 3.7 dicts are OrderPreserving, as such OK
    for episode, segments in speaker_merged_phoneme.items():
        for speaker, samples in segments.items():
            texts = merge_phoneme_of_speaker_samples(samples)

            predicted = text_clf.predict(texts)

            most_common = Counter(predicted).most_common(1)[0][0]  # Get the most common prediction
            string_most_common = PHON_DID_CLS[most_common]
            logger.info(f"Most common prediction for {speaker}: {most_common}, which is {string_most_common}")

            # Save results to collection
            for combination in samples:
                for sample in combination.samples:
                    sample.dialect = string_most_common  # same objects as in meta_data which is saved. I know not pretty, but I was lazy...
                    logger.info(f"NAME: {sample.sample_name}, DID: {string_most_common}")

    write_meta_data(podcast, meta_data)
