import os
import shutil

import h5py
from transformers import Pipeline, Wav2Vec2Processor, pipeline, Wav2Vec2ForCTC

from src.transcription.utils import setup_gpu_device, load_meta_data, get_metadata_path, get_h5_file, write_meta_data, \
    META_WRITE_ITERATIONS
from src.utils.data_points import DatasetDataPoint
from src.utils.logger import get_logger
from src.utils.paths import SCRATCH_PATH, TTS_PODCASTS_PATH

MODEL_AUDIO_PHONEME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

MISSING_PHONEME = "NO_PHONEME"
BATCH_SIZE = 32

logger = get_logger(__name__)


def setup_phoneme_model() -> Pipeline:
    device, torch_dtype = setup_gpu_device()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_AUDIO_PHONEME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_AUDIO_PHONEME)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )


def save_phoneme_results(results: list, samples_to_iterate: list[DatasetDataPoint], start_idx: int,
                         write_to_hdf5: bool, h5: h5py.File) -> list[DatasetDataPoint]:
    for idx, result in enumerate(results):
        phoneme = result["text"].strip()
        if phoneme == "":
            logger.error(f"NO PHONEME TRANSCRIPT GENERATED FOR {samples_to_iterate[start_idx + idx].sample_name}")
            phoneme = MISSING_PHONEME

        if write_to_hdf5:
            h5[samples_to_iterate[start_idx + idx].sample_name].attrs["phoneme"] = phoneme

        samples_to_iterate[start_idx + idx].phoneme = phoneme
        logger.info(f"NAME: {samples_to_iterate[start_idx + idx].sample_name}, PHON: {phoneme}")

    if write_to_hdf5:
        h5.flush()

    return samples_to_iterate


def audio_to_phoneme(podcast: str, write_to_hdf5: bool = True, overwrite_existing_samples: bool = True,
                     copy_from_projects: bool = False) -> None:
    logger.info("Transcribing WAV to phoneme.")

    meta_data, num_samples = load_meta_data(get_metadata_path(podcast))
    if overwrite_existing_samples:
        samples_to_iterate = meta_data
    else:
        samples_to_iterate = [sample for sample in meta_data if sample.phoneme == ""]
        num_samples = len(samples_to_iterate)

    if copy_from_projects:
        h5_file = os.path.join(SCRATCH_PATH, f"{podcast}.hdf5")
        if not os.path.exists(h5_file):
            h5_file = os.path.join(TTS_PODCASTS_PATH, f"{podcast}.hdf5")
            shutil.copy2(h5_file, SCRATCH_PATH)
    else:
        h5_file = get_h5_file(podcast)

    pipe = setup_phoneme_model()

    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        iteration_count = 0
        for start_idx in range(0, num_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_samples)
            # Load batch of audio data
            audio_batch = [h5[samples_to_iterate[i].sample_name][:] for i in range(start_idx, end_idx)]

            # Run phoneme transcription
            results = pipe(audio_batch, batch_size=BATCH_SIZE)

            # Save results to collection
            samples_to_iterate = save_phoneme_results(results, samples_to_iterate, start_idx, write_to_hdf5, h5)

            iteration_count += 1

            # Save progress of transcription in case of failure
            if iteration_count >= META_WRITE_ITERATIONS:
                write_meta_data(podcast, meta_data)
                iteration_count = 0

    write_meta_data(podcast, meta_data)

    if write_to_hdf5 and copy_from_projects:
        shutil.copy2(os.path.join(SCRATCH_PATH, f"{podcast}.hdf5"), TTS_PODCASTS_PATH)


def fix_missing_phoneme(podcast: str, write_to_hdf5: bool = True) -> None:
    """
    Sometimes the chosen phoneme transcription model does not generate a phoneme sequence for the sample and returns
    an empty string. This method will attempt to re-transcribe the samples.

    :param podcast: Podcast name where samples to be re-transcribed
    :param write_to_hdf5: Write the phonemes to h5 file attribute
    :return:
    """
    meta_data, _ = load_meta_data(get_metadata_path(podcast))
    pipe = setup_phoneme_model()
    if len(get_missing_transcriptions(meta_data)) == 0:
        logger.info("No missing phoneme transcription found.")
        return

    logger.info(f"\nAttempting to transcribe samples missing phoneme transcription for podcast {podcast}.\n")

    h5_file = get_h5_file(podcast)
    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        for segment in meta_data:
            if segment.phoneme != MISSING_PHONEME:
                continue

            result = pipe(h5[segment.sample_name][:])
            phoneme = result["text"].strip()

            if phoneme == "":
                logger.error(f"AGAIN NO PHONEME TRANSCRIPT GENERATED FOR {segment.sample_name}.")
                continue

            if write_to_hdf5:
                h5[segment.sample_name].attrs["phoneme"] = phoneme

            segment.phoneme = phoneme
            logger.info(f"NAME: {segment.sample_name}, PHON: {phoneme}")

    if write_to_hdf5:
        h5.flush()

    write_meta_data(podcast, meta_data)


def get_missing_transcriptions(meta_data: list[DatasetDataPoint]) -> list:
    return [entry for entry in meta_data if entry.phoneme == MISSING_PHONEME]
