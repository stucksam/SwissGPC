import json
import os
import random

import h5py
import librosa
import pandas as pd
from datasets import load_dataset
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline

from src.transcription.transcribe_to_phoneme import MODEL_AUDIO_PHONEME
from src.transcription.utils import setup_gpu_device
from src.utils.logger import get_logger

SAMPLING_RATE = 16000

TRAIN_SET_FEMALE_PERCENTAGE = 53.3382 / 100
TRAIN_SET_MALE_PERCENTAGE = 46.6618 / 100
TRAIN_SET_DURATION = 30.0 * 3600  # seconds

TEST_SET_FEMALE_PERCENTAGE = 57.7566 / 100
TEST_SET_MALE_PERCENTAGE = 42.2434 / 100
TEST_SET_DURATION = 4.0 * 3600

VALID_SET_FEMALE_PERCENTAGE = 52.3797 / 100
VALID_SET_MALE_PERCENTAGE = 47.6203 / 100
VALID_SET_DURATION = 4.0 * 3600

TARGET_FEMALE_DURATION = TRAIN_SET_DURATION * TRAIN_SET_FEMALE_PERCENTAGE
TARGET_MALE_DURATION = TRAIN_SET_DURATION * TRAIN_SET_MALE_PERCENTAGE

CV_PARENT_FOLDER = "/home/ubuntu/ma/commonvoice"
CV_EN_PATH = CV_PARENT_FOLDER + "/cv-corpus-21.0-2025-03-14"
CV_DE_PATH = CV_PARENT_FOLDER + "/cv-corpus-19.0-2024-09-13"
DATAPATH_EN = CV_EN_PATH + "/en"
DATAPATH_DE = CV_DE_PATH + "/de"
CLIPS_EN = DATAPATH_EN + "/clips"
CLIPS_DE = DATAPATH_DE + "/clips"

BATCH_SIZE = 32

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

logger = get_logger(__name__)

class_instance = {
    "de": 7,
    "en": 8
}


def _load_h5(split: str, language: str) -> h5py.File:
    h5_file = get_h5_path(split, language)
    return h5py.File(h5_file, "a" if os.path.exists(h5_file) else "w")


def get_h5_path(split: str, language: str) -> str:
    return os.path.join(DATAPATH_EN if language == "en" else DATAPATH_DE, f"{split}_{language}.hdf5")


def get_jsonl_path(split: str, language: str) -> str:
    return os.path.join(DATAPATH_EN if language == "en" else DATAPATH_DE, f"{split}_{language}.jsonl")


def get_distribution_of_property(df: pd.DataFrame, property: str) -> pd.DataFrame:
    # Count each unique age value
    counts = df[property].value_counts().reset_index()
    counts.columns = [property, 'count']  # Rename columns for clarity

    # Add a 'percentage' column
    total_count = counts["count"].sum()
    counts["percentage"] = round((counts["count"] / total_count) * 100, 4)
    return counts


def clean_common_voice(audio_path: str, datapath):
    df = pd.read_csv(f"{datapath}/train.tsv", sep="\t")
    is_null = df[df['gender'].isnull()].reset_index()
    for index, row in is_null.iterrows():
        clip_name = row['path']
        clip_path = f"{audio_path}/{clip_name}"

        if os.path.exists(clip_path):
            logger.info(f"Deleting {clip_name}...")
            os.remove(clip_path)


def get_speakers_by_dataset(split, language) -> set:
    with open(get_jsonl_path(split, language), "r") as f:
        meta_data = [json.loads(line) for line in f]
    return {line["speaker_id"] for line in meta_data}


def create_dataset_of_30_hours(language: str, split: str):
    if language == "en":
        datapath = DATAPATH_EN
        audio_path = CLIPS_EN
    else:
        datapath = DATAPATH_DE
        audio_path = CLIPS_DE

    if split == "train":
        aim_time = TRAIN_SET_DURATION
        fem_aim_time = aim_time * TRAIN_SET_FEMALE_PERCENTAGE
        men_aim_time = aim_time * TRAIN_SET_MALE_PERCENTAGE
    elif split == "valid":
        aim_time = VALID_SET_DURATION
        fem_aim_time = aim_time * VALID_SET_FEMALE_PERCENTAGE
        men_aim_time = aim_time * VALID_SET_MALE_PERCENTAGE
    else:
        aim_time = TEST_SET_DURATION
        fem_aim_time = aim_time * TEST_SET_FEMALE_PERCENTAGE
        men_aim_time = aim_time * TEST_SET_MALE_PERCENTAGE

    dataset_name = split.replace(".tsv", "")
    file_path = os.path.join(datapath, "train.tsv")

    df = pd.read_csv(file_path, sep="\t")
    df_non_null = df[df["gender"].notnull()].reset_index()
    if language == "en":
        df_lang = df_non_null  # too many different accents
    else:
        df_lang = df_non_null[df_non_null["accents"] == "Deutschland Deutsch"]

    df_dur = pd.read_csv(os.path.join(datapath, "clip_durations.tsv"), sep="\t")
    df_merged = pd.merge(df_lang, df_dur, how="inner", left_on="path", right_on="clip")
    if "index" in df_merged.columns.tolist():
        df_merged = df_merged.drop(columns='index')
    df_merged = df_merged.sample(frac=1.0).reset_index(drop=True)

    df_speakers = df_merged.groupby(["client_id", "gender", "age", "accents"], as_index=False)["duration[ms]"].sum()

    current_time = 0.0
    fem_cur_time = 0.0
    men_cur_time = 0.0
    max_time_per_speaker = aim_time * 0.05  # at most 5% should be one speaker

    if "train" in dataset_name:
        used_speakers = set()
    elif "valid" in dataset_name:
        used_speakers = get_speakers_by_dataset("train", language)
    else:
        used_speakers = get_speakers_by_dataset("train", language) | get_speakers_by_dataset("valid", language)

    df_set = pd.DataFrame(columns=df_merged.columns.tolist())
    for index, row in df_speakers.sample(frac=1).iterrows():  # random sampling

        speaker_duration = row["duration[ms]"] / 1000
        if row[
            "client_id"] in used_speakers or speaker_duration > max_time_per_speaker or current_time + speaker_duration > aim_time:
            continue
        if row["gender"] == "male_masculine" and speaker_duration + men_cur_time <= men_aim_time * 1.01:
            men_cur_time += speaker_duration
        elif row["gender"] == "female_feminine" and speaker_duration + fem_cur_time <= fem_aim_time * 1.01:
            fem_cur_time += speaker_duration
        else:
            continue

        df_single = df_merged[df_merged["client_id"] == row["client_id"]]
        df_set = pd.concat([df_set, df_single], ignore_index=True)
        current_time = fem_cur_time + men_cur_time
        if current_time + 60 >= aim_time:
            break

    if "index" in df_set.columns.tolist():
        df_set = df_set.drop(columns='index')

    counts_gender = get_distribution_of_property(df_set, "gender")
    counts_age = get_distribution_of_property(df_set, "age")

    logger.info("Gender Split")
    logger.info(counts_gender)
    logger.info("Age Split")
    logger.info(counts_age)

    h5_file = _load_h5(dataset_name, language)
    json_data = []

    for index, row in df_set.iterrows():
        clip_path = os.path.join(audio_path, row['path'])
        clip_path_wav = clip_path.replace(".mp3", ".wav")

        entry = {"corpus_name": "CommonVoice",
                 "dataset_name": dataset_name,
                 "sample_name": row["path"].replace(".mp3", ""),
                 "class_name": language,
                 "class_nr": class_instance[language],
                 "speaker_id": row["client_id"],
                 "text": row["sentence"],
                 "phonemes": "",
                 "gender": row["gender"],
                 "age": row["age"],
                 "accents": row["accents"]
                 }

        sound = AudioSegment.from_mp3(clip_path)
        sound.export(clip_path_wav, format="wav")  # write segment to os
        speech, _ = librosa.load(clip_path_wav, sr=SAMPLING_RATE)

        _ = h5_file.create_dataset(entry["sample_name"], dtype=float, data=speech)
        h5_file.flush()
        json_data.append(entry)
        os.remove(clip_path_wav)  # keep disk space clean

    # Open the file in write mode
    with open(get_jsonl_path(split, language), "w", encoding="utf-8") as f:
        for entry in json_data:
            # Convert each dictionary to a JSON string and write it as a line
            json_line = json.dumps(entry)
            f.write(json_line + "\n")  # Add newline for the next JSON object


def cv_audio_to_phoneme(split: str, language: str) -> None:
    if language == "en":
        datapath = DATAPATH_EN
    else:
        datapath = DATAPATH_DE

    with open(get_jsonl_path(split, language), "r", encoding="utf-8") as f:
        meta_data = [json.loads(line) for line in f]

    num_samples = len(meta_data)
    device, torch_dtype = setup_gpu_device()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_AUDIO_PHONEME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_AUDIO_PHONEME)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    try:
        with h5py.File(get_h5_path(split, language), "r+") as h5:
            for start_idx in range(0, num_samples, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, num_samples)
                # Load batch of audio data
                audio_batch = [h5[meta_data[i]["sample_name"]][:] for i in range(start_idx, end_idx)]

                results = pipe(audio_batch, batch_size=BATCH_SIZE)
                # Save results
                for idx, result in enumerate(results):
                    phoneme = result["text"].strip()
                    meta_data[start_idx + idx]["phonemes"] = phoneme
                    logger.info(f"NAME: {meta_data[start_idx + idx]['sample_name']}, PHON: {phoneme}")

    except Exception as e:
        logger.error(f"ERROR: {type(e).__name__} with error {str(e)}")

    with open(os.path.join(datapath, f"{split}_{language}_enriched.jsonl"), "wt", encoding="utf-8") as f:
        for entry in meta_data:
            f.write(json.dumps(entry) + "\n")


def filter_common_voice(language: str, split: str):
    ds = load_dataset("mozilla-foundation/common_voice_17_0", language, split=split, use_auth_token=HF_ACCESS_TOKEN)

    # Filter only male/female entries with available audio and gender info
    filtered_ds = ds.filter(lambda x: x["gender"] in {"female_feminine", "male_masculine"})

    # Separate by gender
    female_samples = [x for x in filtered_ds if x["gender"] == "female_feminine"]
    male_samples = [x for x in filtered_ds if x["gender"] == "male_masculine"]

    breakpoint()
    # Shuffle to randomize selection
    random.shuffle(female_samples)
    random.shuffle(male_samples)

    # Helper to accumulate samples until target duration is reached
    def accumulate_samples(samples, target_duration):
        selected = []
        total_duration = 0.0
        for sample in samples:
            duration = librosa.get_duration(path=sample["audio"]["path"])
            if total_duration + duration > target_duration:
                break
            selected.append({
                "sample_name": sample["audio"]["path"].split("/")[-1].replace(".mp3", ""),
                "path": sample["audio"]["path"],
                "duration": duration,
                "speaker": sample["client_id"],
                "text": sample["sentence"],
                "gender": sample["gender"],
                "age": sample["age"],
                "accent": sample["accent"]
            })
            total_duration += duration
        return selected

    # Get balanced datasets
    selected_females = accumulate_samples(female_samples, TARGET_FEMALE_DURATION)
    selected_males = accumulate_samples(male_samples, TARGET_MALE_DURATION)

    merged_df = pd.DataFrame(selected_females + selected_males)
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)

    print(f"Final dataset duration: {merged_df['duration'].sum() / 3600:.2f} hours")
    print(f"Female percentage: {sum([x['duration'] for x in selected_females]) / TRAIN_SET_DURATION * 100:.2f}%")
    print(f"Male percentage: {sum([x['duration'] for x in selected_males]) / TRAIN_SET_DURATION * 100:.2f}%")

    breakpoint()

    h5_file = f"{split}_{language}.hdf5"
    json_data = []
    with h5py.File(h5_file, "a" if os.path.exists(h5_file) else "w") as h5:
        for i, sample in merged_df.iterrow():
            # Save as .wav (resampled)
            clip_path = os.path.join(sample["path"])
            clip_path_wav = clip_path.replace(".mp3", ".wav")

            entry = {"corpus_name": "CommonVoice",
                     "dataset_name": split,
                     "sample_name": sample["sample_name"],
                     "classname": language,
                     "duration": sample["duration"],
                     "class_nr": class_instance[language],
                     "output_len": -1,
                     "speaker_id": sample["client_id"],
                     "text": sample["sentence"],
                     "phonemes": "",
                     "gender": sample["gender"],
                     "age": sample["age"],
                     "accent": sample["accent"]
                     }

            sound = AudioSegment.from_mp3(clip_path)
            sound.export(clip_path_wav, format="wav")  # write segment to os
            speech, _ = librosa.load(clip_path_wav, sr=SAMPLING_RATE)
            _ = h5.create_dataset(entry["sample_name"], dtype=float, data=speech)
            h5.flush()

            json_data.append(entry)
            os.remove(clip_path_wav)  # keep disk space clean

            # Open the file in write mode
    with open(f"cv_{language}_{split}.jsonl", "w", encoding="utf-8") as f:
        for entry in json_data:
            # Convert each dictionary to a JSON string and write it as a line
            json_line = json.dumps(entry)
            f.write(json_line + "\n")  # Add newline for the next JSON object
