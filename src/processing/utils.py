import os

from src.utils.paths import CLUSTER_PROJECTS_PATH, CLUSTER_PROJECTS_TTS

CLUSTER_PROJECTS_DERI = os.path.join(CLUSTER_PROJECTS_PATH, "deri_tts")

SNF_DATASET_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "snf_train_set")
SWISSDIAL_DATASET_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "swissdial")

SWISSDIAL_CANTON_TO_DIALECT = {
    "gr": "Graubünden",
    "lu": "Innerschweiz",
    "zh": "Zürich",
    "vs": "Wallis",
    "sg": "Ostschweiz",
    "be": "Bern",
    "bs": "Basel",
    "ag": "Zürich",
}
