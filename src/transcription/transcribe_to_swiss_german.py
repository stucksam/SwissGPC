import os

import h5py
import torch
from transformers import T5Tokenizer, PreTrainedModel, T5ForConditionalGeneration

from src.transcription.utils import DIALECT_TO_TAG, MISSING_TEXT, load_meta_data, get_metadata_path, get_h5_file, \
    setup_gpu_device, META_WRITE_ITERATIONS, write_meta_data
from src.utils.data_points import DatasetDataPoint
from src.utils.logger import get_logger
from src.utils.paths import MODEL_PATH

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

MODEL_PATH_DE_CH = f"{MODEL_PATH}/de_to_ch_large_2"
MODEL_T5_TOKENIZER = "google/t5-v1_1-large"
BATCH_SIZE = 16

NO_CH_TEXT = "NO_CH_TEXT"

logger = get_logger(__name__)


def run_ch_de_batch(start_idx: int, batch_size: int, num_samples: int, samples_to_iterate: list[DatasetDataPoint],
                    tokenizer: T5Tokenizer, model: PreTrainedModel, device: str) -> list:
    end_idx = min(start_idx + batch_size, num_samples)
    batch_texts = [f"[{DIALECT_TO_TAG[samples_to_iterate[i].dialect]}]: {samples_to_iterate[i].de_text}" for i in
                   range(start_idx, end_idx)]  # Extract batched texts from meta_data

    # Tokenize the batch of sentences
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=400)

    # Move input tensors to the device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate translations
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, max_length=400, attention_mask=attention_mask, num_beams=5,
                                    num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    batch_translations = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    batch_translations = [x.replace("Ä ", "Ä").replace("Ü ", "Ü").replace("Ö ", "Ö").strip() for x in
                          batch_translations]
    return batch_translations


def save_ch_de_results(batch_translations: list, samples_to_iterate: list[DatasetDataPoint], start_idx: int,
                       write_to_hdf5: bool, h5: h5py.File) -> list[DatasetDataPoint]:
    for idx, ch_text in enumerate(batch_translations):
        if ch_text == "":
            logger.error(f"NO SWISS GERMAN TRANSCRIPT GENERATED FOR {samples_to_iterate[start_idx + idx].sample_name}")
            ch_text = MISSING_TEXT

        if write_to_hdf5:
            h5[samples_to_iterate[start_idx + idx].sample_name].attrs["ch_text"] = ch_text

        samples_to_iterate[start_idx + idx].ch_text = ch_text
        logger.info(f"DE: {samples_to_iterate[start_idx + idx].de_text}, CH: {ch_text}")

    if write_to_hdf5:
        h5.flush()

    return samples_to_iterate


def transcribe_de_to_ch(podcast: str, write_to_hdf5: bool = True, overwrite_existing_samples: bool = True) -> None:
    """
    Instead of directly transcribing audio to CH-DE we chose the approach of first transcribing it to Standard German
    and then translate it to Swiss German.
    :param podcast:
    :param write_to_hdf5:
    :param overwrite_existing_samples:
    :return:
    """
    logger.info("Transcribing German text to Swiss German text.")
    meta_data, _ = load_meta_data(get_metadata_path(podcast))
    meta_data_non_de = [sample for sample in meta_data if sample.dialect != "Deutschland"]
    num_samples = len(meta_data_non_de)

    if overwrite_existing_samples:
        samples_to_iterate = meta_data_non_de
    else:
        samples_to_iterate = [sample for sample in meta_data_non_de if sample.ch_text == ""]
        num_samples = len(samples_to_iterate)

    h5_file = get_h5_file(podcast)
    device, _ = setup_gpu_device()

    model = T5ForConditionalGeneration.from_pretrained(os.path.join(MODEL_PATH_DE_CH, "best-model"))
    tokenizer = T5Tokenizer.from_pretrained(MODEL_T5_TOKENIZER)
    tokenizer.add_tokens(["Ä", "Ö", "Ü"])

    model.to(device)
    model.eval()

    with h5py.File(h5_file, "r+" if write_to_hdf5 else "r") as h5:
        iteration_count = 0
        for start_idx in range(0, num_samples, BATCH_SIZE):
            batch_translations = run_ch_de_batch(start_idx, BATCH_SIZE, num_samples, samples_to_iterate, tokenizer,
                                                 model, device)

            # Save results to collection
            samples_to_iterate = save_ch_de_results(batch_translations, samples_to_iterate, start_idx, write_to_hdf5,
                                                    h5)
            iteration_count += 1

            if iteration_count >= META_WRITE_ITERATIONS:
                write_meta_data(podcast, meta_data)
                iteration_count = 0

    for sample in meta_data:
        if sample.dialect == "Deutschland":
            sample.ch_text = NO_CH_TEXT

    write_meta_data(podcast, meta_data)
