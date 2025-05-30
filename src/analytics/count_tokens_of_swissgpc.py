import os
from collections import Counter

import seaborn as sns
import spacy
from matplotlib import pyplot as plt

from src.transcription.utils import load_meta_data, get_metadata_path
from src.utils.logger import get_logger
from src.utils.paths import PODCAST_AUDIO_FOLDER

nlp = spacy.load('de_core_news_sm')


logger = get_logger(__name__)


def swissgpc_token_distribution():
    token_path = os.path.join(PODCAST_AUDIO_FOLDER, "Tokens")
    txt_files = [f for f in os.listdir(token_path) if f.endswith('.txt')]
    token_counts = []
    for file in txt_files:
        with open(os.path.join(token_path, file), "rt", encoding="utf-8") as meta_file:
            for line in meta_file:
                split_line = line.replace('\n', '').split('\t')
                try:
                    # tokens = int(float(split_line[2]))  # if you want the time
                    tokens = int(split_line[-1])

                except ValueError:
                    continue
                if tokens >= 80:
                    logger.info(f"{file}: {split_line[0]} -> tokens: {tokens}")
                token_counts.append(tokens)

    # Count occurrences of each token length
    token_distribution = Counter(token_counts)
    logger.info(token_distribution)
    # Prepare data for plotting
    x = sorted(token_distribution.keys())  # Unique token counts
    y = [token_distribution[t] for t in x]  # Corresponding frequencies
    plot_token_distro(x, y, p_type="SwissGPC")


def run_token_analysis(sentences: list[str], p_type: str) -> None:
    # Tokenize with spaCy
    token_counts = [len(nlp(sentence)) for sentence in sentences]

    # Count occurrences of each token length
    token_distribution = Counter(token_counts)

    # Prepare data for plotting
    x = sorted(token_distribution.keys())  # Unique token counts
    y = [token_distribution[t] for t in x]  # Corresponding frequencies
    plot_token_distro(x, y, p_type=p_type)


def plot_token_distro(x, y, t_type: str = "spaCy", p_type: str = "SRF") -> None:
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y)
    # plt.xlabel('Duration in seconds', fontsize=14)
    plt.xlabel('Number of Tokens', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    # plt.title(f"{p_type} Time Distribution", fontsize=16)
    plt.title(f"{p_type} Token Distribution with {t_type}", fontsize=16)
    # plt.xticks(ticks=range(0, 15), rotation=45)  # Adjust tick step if necessary
    plt.xticks(ticks=range(0, 66, 2), rotation=45)  # Adjust tick step if necessary
    plt.xlim(0, 66)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Avoid label overlapping
    # plt.savefig(f"time_distribution_{p_type}_{t_type}.png")
    plt.savefig(f"token_distribution_{p_type}_{t_type}.png")
    plt.show()


def calc_token_per_podcast():
    txt_files = [f for f in os.listdir(PODCAST_AUDIO_FOLDER) if f.endswith('.txt')]
    for file in txt_files:
        meta_data, _ = load_meta_data(f"{PODCAST_AUDIO_FOLDER}/{file}", load_as_dialect=False)
        podcast = file.replace(".txt", "")
        tokens = [len(nlp(sample.de_text)) for sample in meta_data]
        with open(get_metadata_path(podcast).replace(podcast, f"tokens_{podcast}"), "wt", encoding="utf-8") as f:
            for i, line in enumerate(meta_data):
                line = line.to_string().replace("\n", f"\t{tokens[i]}\n")
                f.write(line)
