import os

import h5py
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.transcription.utils import setup_gpu_device, get_h5_file, load_meta_data, get_metadata_path, \
    META_WRITE_ITERATIONS, write_meta_data, MISSING_TEXT
from src.utils.data_points import DatasetDataPoint
from src.utils.logger import get_logger

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

MODEL_WHISPER_v3 = "openai/whisper-large-v3"
BATCH_SIZE = 32

logger = get_logger(__name__)


def setup_german_transcription_model():
    device, torch_dtype = setup_gpu_device()
    # could do more finetuning under "or more control over the generation parameters, use the model + processor API directly: "
    # in https://github.com/huggingface/distil-whisper/blob/main/README.md
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_WHISPER_v3, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_WHISPER_v3)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "german"}
    )


def save_de_transcribe_results(results: list, samples_to_iterate: list[DatasetDataPoint], start_idx: int,
                               write_to_hdf5: bool, h5: h5py.File) -> list[DatasetDataPoint]:
    for idx, result in enumerate(results):
        text = result["text"].strip()
        if text == "" or text == "...":
            logger.error(f"NO GERMAN TRANSCRIPT GENERATED FOR {samples_to_iterate[start_idx + idx].sample_name}")
            text = MISSING_TEXT

        if write_to_hdf5:
            h5[samples_to_iterate[start_idx + idx].sample_name].attrs["de_text"] = text

        samples_to_iterate[start_idx + idx].de_text = text
        logger.info(f"NAME: {samples_to_iterate[start_idx + idx].sample_name}, TXT: {text}")

    if write_to_hdf5:
        h5.flush()

    return samples_to_iterate


def transcribe_audio_to_german(podcast: str, write_to_hdf5: bool = True,
                               overwrite_existing_samples: bool = True) -> None:
    # You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
    logger.info("Transcribing to WAV to German")
    meta_data, num_samples = load_meta_data(get_metadata_path(podcast))

    if overwrite_existing_samples:
        samples_to_iterate = meta_data
    else:
        samples_to_iterate = [sample for sample in meta_data if sample.de_text == ""]
        num_samples = len(samples_to_iterate)

    h5_file = get_h5_file(podcast)
    pipe = setup_german_transcription_model()

    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        iteration_count = 0
        for start_idx in range(0, num_samples, BATCH_SIZE):
            # Define the batch range
            end_idx = min(start_idx + BATCH_SIZE, num_samples)
            logger.debug(f"Transcribing samples {start_idx} to {end_idx}...")
            # Load batch of audio data
            audio_batch = [h5[samples_to_iterate[i].sample_name][:] for i in range(start_idx, end_idx)]

            # Perform transcription
            results = pipe(audio_batch, batch_size=BATCH_SIZE)

            # Save results to collection
            samples_to_iterate = save_de_transcribe_results(results, samples_to_iterate, start_idx, write_to_hdf5, h5)

            iteration_count += 1

            # Save progress of transcription in case of failure
            if iteration_count >= META_WRITE_ITERATIONS:
                write_meta_data(podcast, meta_data)
                iteration_count = 0

    write_meta_data(podcast, meta_data)


def fix_long_german_segments(podcast: str, write_to_hdf5: bool = True):
    """
    Sometimes whisper returns very random de-texts, containing only repetitions of the same world like "erst, erst, erst,
    erst, erst,..." etc. These segments are detected (generally len of > 390 characters) and re-run through whisper. Let
    them purposefully run through without batching
    :return:
    """
    meta_data, _ = load_meta_data(get_metadata_path(podcast))
    pipe = setup_german_transcription_model()
    long_segments = get_missing_transcriptions(meta_data)
    if len(long_segments) == 0:
        logger.info("No missing phoneme transcription found.")
        return

    logger.info(f"\nAttempting to transcribe samples missing German transcription for podcast {podcast}.\n")

    h5_file = get_h5_file(podcast)
    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        for segment in long_segments:

            result = pipe(h5[segment.sample_name][:])
            text = result["text"].strip()

            if len(text) > 390 or text == "...":
                logger.error(f"AGAIN NO OR ERRONEOUS GERMAN TRANSCRIPT GENERATED FOR {segment.sample_name}")
                continue

            if write_to_hdf5:
                h5[segment.sample_name].attrs["de_text"] = text

            segment.de_text = text
            logger.info(f"NAME: {segment.sample_name}, DE-TXT: {text}")

    if write_to_hdf5:
        h5.flush()

    write_meta_data(podcast, meta_data)


def get_missing_transcriptions(meta_data: list[DatasetDataPoint]) -> list:
    return [entry for entry in meta_data if len(entry.de_text) > 390 or entry.de_text in [MISSING_TEXT, "..."]]
