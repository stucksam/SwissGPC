import os

SCRATCH_PATH = "/scratch"
MODEL_PATH = "models/"
CLUSTER_PROJECTS_PATH = "/cluster/projects/"

CLUSTER_PROJECTS_TTS = os.path.join(CLUSTER_PROJECTS_PATH, "TTS-Swiss-German")
TTS_PODCASTS_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "audio_transcribed_new")
TTS_TRAINING_SUBSETS_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "audio_subsets")
TTS_RAW_AUDIO_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "audio_raw")

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.getcwd()
PODCAST_METADATA_FOLDER = os.path.join(PROJECT_DIR, "src", "download", "metadata")
PODCAST_AUDIO_FOLDER = os.path.join(PROJECT_DIR, "audio")
# PODCAST_AUDIO_FOLDER = os.path.join(CLUSTER_PROJECTS_TTS, "audio_raw")
