import argparse

import yaml

from src.classification.dialect_classifier import dialect_identification_naive_bayes_majority_voting
from src.download.download_from_srf import download_srf_podcast_audio, download_srf_podcast_metadata
from src.download.download_from_yt import download_yt_podcast_audio, download_yt_podcast_metadata
from src.processing.move_audio_to_dialect import move_podcast_to_dialect
from src.segmentation.segmentation import diarize_and_segment_podcast
from src.synthesis.mel_spectrogram import create_mel_spectrogram
from src.transcription.transcribe_to_phoneme import audio_to_phoneme
from src.transcription.transcribe_to_swiss_german import transcribe_de_to_ch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(config_path):
    logger.info("Started")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    source = config["source"].lower()
    assert source in ["youtube", "srf"], "Values for 'source' must be either 'youtube' or 'srf'"

    podcast_name = config["podcast_name"]
    write_to_hdf5 = config["write_attrs_to_hdf5"]
    logger.info(f"Transcribing Podcast {podcast_name} from {source}.")

    # Step 1: Download Audio
    if config["steps"]["download"]:
        if source == "youtube":
            download_yt_podcast_metadata(podcast_name, config["youtube_url"])
            download_yt_podcast_audio(podcast_name)
        else:
            download_srf_podcast_metadata(podcast_name)
            download_srf_podcast_audio(podcast_name)

    # Step 2: Speaker Diarization & German Transcription & Segmentation
    if config["steps"]["diarization"] or config["steps"]["segmentation"]:
        diarize_and_segment_podcast(podcast_name, config["steps"]["diarization"], config["steps"]["segmentation"], copy_to_projects=True)

    # Step 3: Phoneme Transcription
    if config["steps"]["phon_transcription"]:
        audio_to_phoneme(podcast_name, write_to_hdf5, overwrite_existing_samples=False, copy_from_projects=True)

    # Step 4: Dialect Identification
    if config["steps"]["dialect_classification"]:
        dialect_identification_naive_bayes_majority_voting(podcast_name)

    # Step 5: Swiss German Text Generation
    if config["steps"]["ch_transcription"]:
        transcribe_de_to_ch(podcast_name, write_to_hdf5)

    # Step 6: Mel-Spectrogram Generation
    if config["steps"]["mel_spectrogram"]:
        create_mel_spectrogram(podcast_name)

    # Step 7: Move podcast based h5 into central dialect h5
    if config["steps"]["move_into_dialect_h5"]:
        move_podcast_to_dialect(podcast_name)

    logger.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
