class DatasetDataPoint:
    # use an object instead of a dict since the trainer removes all columns that do not match the signature
    def __init__(
            self,
            sample_name: str,
            duration: float,
            track_start: float,
            track_end: float,
            track_id: int,
            speaker_id: str,
            de_text: str = "",
            phoneme: str = "",
            dialect: str = "",
            ch_text: str = "",
    ):
        self.dataset_name = ""
        self.sample_name = sample_name
        self.duration = duration
        self.track_start = track_start
        self.track_end = track_end
        self.track_id = track_id
        self.speaker_id = speaker_id
        self.de_text = de_text
        self.phoneme = phoneme
        self.dialect = dialect
        self.ch_text = ch_text

        split_name = self.sample_name.split("_")
        if len(split_name) > 2:
            self.orig_episode_name = '_'.join(split_name[:-1])
        else:
            self.orig_episode_name = split_name[0]

    @staticmethod
    def load_single_datapoint(split_properties: list):
        sample_name = split_properties[0]
        track_id = int(split_properties[1])
        duration = float(split_properties[2])
        track_start = float(split_properties[3])
        track_end = float(split_properties[4])
        speaker = split_properties[5]
        return DatasetDataPoint(
            sample_name=sample_name,
            duration=duration,
            track_start=track_start,
            track_end=track_end,
            track_id=track_id,
            speaker_id=speaker,
            de_text=split_properties[6] if len(split_properties) > 6 else "",
            phoneme=split_properties[7] if len(split_properties) > 7 else "",
            dialect=split_properties[8] if len(split_properties) > 8 else "",
            ch_text=split_properties[9] if len(split_properties) > 9 else ""
        )

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, dataset_name):
        self._dataset_name = dataset_name

    def to_string(self):
        to_string = f"{self.sample_name}\t{self.track_id}\t{self.duration}\t{self.track_start}\t{self.track_end}\t{self.speaker_id}"

        if self.de_text:
            to_string += f"\t{self.de_text}"
        if self.phoneme:
            to_string += f"\t{self.phoneme}"
        if self.dialect:
            to_string += f"\t{self.dialect}"
        if self.ch_text:
            to_string += f"\t{self.ch_text}"

        to_string += "\n"
        return to_string

    def convert_to_dialect_datapoint(self):
        return DialectDataPoint(
            self.dataset_name,
            self.sample_name,
            self.duration,
            self.speaker_id,
            self.dialect,
            self.de_text
        )


class DialectDataPoint:
    def __init__(
            self,
            dataset_name: str,
            sample_name: str,
            duration: float,
            speaker_id: str,
            dialect: str,
            de_text: str,
    ):
        self.dataset_name = dataset_name
        self.sample_name = sample_name
        self.duration = float(duration)
        self.speaker_id = speaker_id
        self.dialect = dialect
        self.de_text = de_text

        split_name = self.sample_name.split("_")
        if len(split_name) > 2:
            self.orig_episode_name = '_'.join(split_name[:-1])
        else:
            self.orig_episode_name = split_name[0]

    @staticmethod
    def number_of_properties():
        return 6

    @staticmethod
    def load_single_datapoint(split_properties: list):
        return DialectDataPoint(
            dataset_name=split_properties[0],
            sample_name=split_properties[1],
            duration=split_properties[2],
            speaker_id=split_properties[3],
            dialect=split_properties[4],
            de_text=split_properties[5]
        )

    def to_string(self):
        return f"{self.dataset_name}\t{self.sample_name}\t{self.duration}\t{self.speaker_id}\t{self.dialect}\t{self.de_text}\n"
