import json
import os
import shutil
from typing import TextIO

import h5py
import librosa
import whisperx
from pydub import AudioSegment

from src.download.utils import PODCAST_AUDIO_FOLDER, load_podcast_metadata_from_csv, get_podcast_path
from src.segmentation.filter_strategies import filter_segments_using_strats
from src.utils.logger import get_logger
from src.utils.paths import SCRATCH_PATH, TTS_PODCASTS_PATH, MODEL_PATH, TTS_RAW_AUDIO_PATH

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

SAMPLING_RATE = 16000

logger = get_logger(__name__)


def _load_sample(path: str) -> AudioSegment:
    return AudioSegment.from_file(path)


def _load_txt_meta(podcast: str) -> [TextIO, set]:
    """
    Built as sample_name -> track_id -> duration -> track_start -> track_end -> speaker -> de_text
    :param podcast:
    :return:
    """
    metadata_txt = os.path.join(PODCAST_AUDIO_FOLDER, f'{podcast}.txt')
    already_processed = set()
    if os.path.exists(metadata_txt):
        with open(metadata_txt, 'rt', encoding='utf-8') as meta_data_file:
            already_processed = set([x.split('\t')[0] for x in meta_data_file.readlines()])
        return open(metadata_txt, 'at', encoding='utf-8'), already_processed
    else:
        return open(metadata_txt, 'wt', encoding='utf-8'), already_processed


def get_episode_path(podcast_path: str, ep_id: str) -> str:
    return os.path.join(podcast_path, f"{ep_id}.mp3")


def get_diarized_file_path(podcast_path: str, ep_id: str) -> str:
    return os.path.join(podcast_path, f"{ep_id}.json")


def get_hdf5_file(podcast: str, copy_to_projects: bool = False) -> str:
    if copy_to_projects:
        return os.path.join(SCRATCH_PATH, f"{podcast}.hdf5")
    else:
        return os.path.join(PODCAST_AUDIO_FOLDER, f"{podcast}.hdf5")


def diarize_and_segment_podcast(podcast: str, do_diarization: bool = True, do_segmentation: bool = True,
                                copy_to_projects: bool = False) -> None:
    df = load_podcast_metadata_from_csv(podcast)
    podcast_path = get_podcast_path(podcast)

    if do_diarization:
        for i, row in df.iterrows():
            ep_id = row["id"]
            episode_path = get_episode_path(podcast_path, ep_id)
            diarized_file_path = get_diarized_file_path(podcast_path, ep_id)

            if not os.path.exists(episode_path):
                logger.error(f"Episode {ep_id} does not exist in {podcast} audio.")
                continue

            elif os.path.exists(diarized_file_path):
                logger.info(f"Episode {ep_id} has already been diarized.")
                continue

            else:
                diarize_episode(podcast_path, ep_id)

    if do_segmentation:
        if copy_to_projects:
            podcast_path = os.path.join(TTS_RAW_AUDIO_PATH, podcast)
            shutil.copytree(podcast_path, os.path.join(SCRATCH_PATH, podcast), dirs_exist_ok=True)
            podcast_path = os.path.join(SCRATCH_PATH, podcast)

        h5_file_path = get_hdf5_file(podcast, copy_to_projects)
        with h5py.File(h5_file_path, "a" if os.path.exists(h5_file_path) else "w") as h5:
            for i, row in df.iterrows():
                ep_id = row["id"]
                diarized_file_path = get_diarized_file_path(podcast_path, ep_id)

                if not os.path.exists(diarized_file_path):
                    logger.error(f"Episode {ep_id} diarization does not exist in {podcast} folder.")
                    continue
                else:
                    cut_episode_into_segments(podcast, ep_id, h5, copy_to_projects=copy_to_projects)

        if copy_to_projects:
            shutil.copy2(os.path.join(SCRATCH_PATH, f"{podcast}.hdf5"), TTS_PODCASTS_PATH)
            shutil.copy2(os.path.join(PODCAST_AUDIO_FOLDER, f"{podcast}.txt"), TTS_PODCASTS_PATH)


def diarize_episode(podcast: str, ep_id: str) -> list:
    device = "cuda"
    batch_size = 32  # reduce if low on GPU mem
    compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
    podcast_path = get_podcast_path(podcast)
    episode_path = get_episode_path(podcast_path, ep_id)

    logger.info(f"Diarizing episode {ep_id}")

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v3", device, language="de", compute_type=compute_type,
                                download_root=MODEL_PATH)

    audio = whisperx.load_audio(episode_path)
    result = model.transcribe(audio, batch_size=batch_size, chunk_size=15, language="de")
    # logger.info(result["segments"])  # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # logger.info(result["segments"])  # after alignment
    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_ACCESS_TOKEN, device=device)

    # add min/max number of speakers if known
    # diarize_segments = diarize_model(audio)
    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    with open(get_diarized_file_path(podcast_path, ep_id), "w", encoding='utf8') as f:
        json.dump(result["segments"], f, indent=4)

    return result["segments"]


def cut_episode_into_segments(podcast: str, episode_id: str, h5: h5py.File, save_filtered_output: bool = False,
                              save_cuts_as_mp3: bool = False, copy_to_projects: bool = False) -> None:
    logger.info(f"Segmenting episode {episode_id}")

    podcast_path = get_podcast_path(podcast)

    if copy_to_projects:
        podcast_path = os.path.join(SCRATCH_PATH, podcast)

    episode_path = get_episode_path(podcast_path, episode_id)
    diarized_file_path = get_diarized_file_path(podcast_path, episode_id)

    metadata_txt, already_processed = _load_txt_meta(podcast)
    start_id = 1000  # enables better id assignment with 1 being lower than 10 in files due to 1001 and 1010

    with open(diarized_file_path, "r", encoding='utf8') as f:
        segments = json.load(f)

    filtered_segments = filter_segments_using_strats(segments)

    if save_filtered_output:
        with open(diarized_file_path.replace(".json", "_merged.json"), "w", encoding='utf8') as f:
            json.dump(filtered_segments, f, indent=4)

    audio = _load_sample(episode_path)

    for i, segment in enumerate(filtered_segments):
        segment_id = start_id + i
        segment_name = f"{episode_id}_{segment_id}"
        segment_path = os.path.join(podcast_path, f"{segment_name}.wav")

        if segment_name in already_processed:
            logger.debug(f"Cut {segment_id} of episode {episode_id} is already cut.")
            continue

        duration = segment["end"] - segment["start"]
        start_ms = segment["start"] * 1000
        end_ms = segment["end"] * 1000

        audio_segment = audio[start_ms: end_ms]
        audio_segment.export(segment_path, format="wav")  # write segment to os for librosa to load in target SR
        if save_cuts_as_mp3:
            audio_segment.export(segment_path.replace("wav", "mp3"), format="mp3")  # write segment to os

        duration = round(duration, 4)
        speaker = segment['speaker']
        track_start = round(segment['start'], 4)
        track_end = round(segment['end'], 4)
        de_text = segment["text"].strip()

        speech, _ = librosa.load(segment_path, sr=SAMPLING_RATE)

        h5_entry = h5.create_dataset(segment_name, dtype=float, data=speech)
        h5_entry.attrs["dataset_name"] = podcast
        h5_entry.attrs["speaker"] = speaker
        h5_entry.attrs["duration"] = duration
        h5_entry.attrs["track_start"] = track_start
        h5_entry.attrs["track_end"] = track_end
        h5_entry.attrs["de_text"] = de_text
        h5.flush()

        metadata_txt.write(
            f"{segment_name}\t{segment_id}\t{duration}\t{track_start}\t{track_end}\t{speaker}\t{de_text}\n")

        os.remove(segment_path)  # keep disk space clean

    metadata_txt.close()
