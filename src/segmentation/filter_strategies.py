import math
from collections import Counter

from src.utils.logger import get_logger

MIN_SAMPLE_DURATION = 2.0
MAX_SAMPLE_DURATION = 15.0

MAX_SILENCE_DURATION = 2.0

logger = get_logger(__name__)


def is_audio_complex(segment: dict) -> bool:
    if "speaker" not in segment:
        return True

    words = segment["words"]
    speakers = [word["speaker"] for word in words if "speaker" in word]
    speaker_counts = dict(Counter(speakers))
    if len(speaker_counts) == 0:
        return True
    elif len(speaker_counts) == 1:
        return False
    else:
        speaker_counts = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)
        if speaker_counts[0][1] / len(words) < 0.75:
            return True

    return False


def strat_simple_segmentation(segments: list) -> list:
    """
    Strategy "simple" segments the audio based on the initial whisperx diarization without changing anything on the
    samples. Only filtering performed will be the removal of samples with less than MIN_SAMPLE_DURATION seconds or more
     than MAX_SAMPLE_DURATION. Audio considered complex by whisperx will be removed
    :return:
    """
    filtered_segments = []
    for i, segment in enumerate(segments):
        duration = segment["end"] - segment["start"]
        if duration < MIN_SAMPLE_DURATION or duration > MAX_SAMPLE_DURATION or is_audio_complex(segment):
            continue

        filtered_segments.append(segment)

    return filtered_segments


def strat_merge_to_min_amount_of_seconds(segments: list) -> list:
    def check_following_segments(orig_segment: dict, compare_segment: dict) -> dict | None:
        # Don't merge non-neighbouring segments
        if orig_segment["speaker"] != compare_segment["speaker"]:
            filtered_segments.append(orig_segment)
            return None

        # We don't want to increase size of already large segments
        elif compare_segment["end"] - compare_segment["start"] > MAX_SAMPLE_DURATION:
            filtered_segments.append(orig_segment)
            return None

        # Don't merge large silences inbetween segments
        elif compare_segment["start"] - orig_segment["end"] > MAX_SILENCE_DURATION:
            filtered_segments.append(orig_segment)
            return None

        else:
            to_skip.add(i + j + 1)

            return {
                "start": orig_segment["start"],
                "end": compare_segment["end"],
                "text": f"{orig_segment['text']} {compare_segment['text']}",
                "words": orig_segment["words"] + compare_segment["words"],
                "speaker": orig_segment["speaker"]
            }

    skipped_counter = 0
    filtered_segments = []
    to_skip = set()
    for i, segment in enumerate(segments):

        if i in to_skip or "speaker" not in segment:
            continue

        if i == len(segments) - 1:
            filtered_segments.append(segment.copy())
            break

        duration = segment["end"] - segment["start"]

        if duration < MIN_SAMPLE_DURATION:
            following_segments = segments[i + 1:]

            checked_segment = segment.copy()
            for j, compare_track in enumerate(following_segments):

                if "speaker" not in compare_track:
                    filtered_segments.append(checked_segment)
                    to_skip.add(i + j + 1)
                    skipped_counter += 1
                    break

                checked_segment = check_following_segments(checked_segment, compare_track)

                # No change to segment made, we will ignore it
                if checked_segment is None:
                    break

                # If we reached a segment of min sample dur length stop merging of samples.
                elif checked_segment["end"] - checked_segment["start"] >= MIN_SAMPLE_DURATION:
                    filtered_segments.append(checked_segment)
                    break


        else:
            filtered_segments.append(segment)

    logger.debug(f"Skipped {skipped_counter} segments due to missing speaker.")
    logger.debug(f"Skipped {len(to_skip)} segments as they were merged with another segment.")
    logger.debug(f"Filtered {len(filtered_segments)} segments compared to original {len(segments)} segments.")
    return filtered_segments


def strat_cut_to_max_amount_of_seconds(segments: list) -> list:
    skipped_counter = 0
    cut_counter = 0
    filtered_segments = []
    to_skip = set()
    for i, segment in enumerate(segments):

        if i in to_skip or "speaker" not in segment:
            continue

        duration = segment["end"] - segment["start"]

        if duration > MAX_SAMPLE_DURATION:
            checked_segment = segment.copy()
            number_of_segments = math.ceil(duration / MAX_SAMPLE_DURATION)
            min_new_segment_dur = duration / number_of_segments
            new_segments = [{} for _ in range(number_of_segments)]

            current_segment_index = 0

            for j, word in enumerate(checked_segment["words"]):
                # Alignment somtimes fails resulting in NO timestamps or other information. If it is not last word
                # we just continue and check next word.
                if "end" not in word or "start" not in word:
                    if j != len(checked_segment["words"]) - 1:

                        # if timestamp of is missing inside sentence add it, if at the beginning (new segment) do not
                        # as we can not pinpoint exact mention of word on timeline
                        if "text" in new_segments[current_segment_index]:
                            new_segments[current_segment_index]["text"] += f" {word['word']}"
                            new_segments[current_segment_index]["words"].append(word)

                        continue

                    else:
                        break

                if "start" not in new_segments[current_segment_index]:
                    new_segments[current_segment_index] = {
                        "start": word["start"],
                        "end": word["end"],
                        "text": word["word"],
                        "words": [word],
                        "speaker": checked_segment["speaker"],
                        "is_cut": True
                    }
                else:
                    if word["end"] - new_segments[current_segment_index]["start"] >= min_new_segment_dur:
                        current_segment_index += 1

                        # it often happens that the last word is incorrectly assigned, sometimes with a duration of only
                        # 1-2 seconds, but it occurs 6-7 seconds after the previous word
                        if j == len(checked_segment["words"]) - 1:
                            break
                    else:
                        new_segments[current_segment_index]["end"] = word["end"]
                        new_segments[current_segment_index]["text"] += f" {word['word']}"
                        new_segments[current_segment_index]["words"].append(word)

            cut_counter += number_of_segments
            # only append segments that actually are filled
            filtered_segments.extend([seg for seg in new_segments if seg != {}])

        else:
            filtered_segments.append(segment)

    logger.debug(f"Skipped {skipped_counter} segments due to missing speaker.")
    logger.debug(f"Cut into {cut_counter} new segments.")
    logger.debug(f"Filtered {len(filtered_segments)} segments compared to original {len(segments)} segments.")
    return filtered_segments


def filter_segments_using_strats(segments: list) -> list:
    segments = strat_merge_to_min_amount_of_seconds(segments)
    segments = strat_cut_to_max_amount_of_seconds(segments)
    segments = strat_simple_segmentation(segments)
    return segments
