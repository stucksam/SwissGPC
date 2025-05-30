import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class_nr_map = {"Bern": 6, "Wallis": 2, "Basel": 5, "Graubünden": 4, "Ostschweiz": 3, "Zürich": 0, "Innerschweiz": 1,
                "Deutschland": 7, "English": 8}


def load_dataset(path):
    dataset = []
    with open(path, "rt", encoding="utf-8") as ifile:
        for line in ifile:
            dataset.append(json.loads(line))
    return dataset


def concat_per_speaker(dataset, remove_blank=False, n=3):
    samples_per_speaker = defaultdict(list)
    for entry in dataset:
        samples_per_speaker[entry["speaker_id"]].append(entry)

    text_data, labels = [], []
    for speaker_id, samples in tqdm(samples_per_speaker.items()):
        label = samples[0]["class_nr"]
        if len(samples) < n:
            continue
        for i in range(len(samples)):
            rand_sample = random.sample(samples, k=n)
            text = " ".join([x["phonemes"] for x in rand_sample])
            if remove_blank:
                text = text.replace(" ", "")
            text_data.append(text)
            labels.append(label)
    return text_data, labels


def run_double_loop(train_set, train_samples, remove_blank, text_clf=None):
    X_train, y_train = concat_per_speaker(train_set, remove_blank, n=train_samples)

    if text_clf is None:
        text_clf = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 5), analyzer="char", sublinear_tf=True, min_df=10,
                                          max_features=150_000)),
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", SGDClassifier(loss="hinge", warm_start=True, max_iter=10000, tol=1e-4, verbose=0, n_jobs=8)),
            ]
            , verbose=True
        )
    else:
        text_clf = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**text_clf.get_params()["steps"][0][1].get_params())),
                ("scaler", StandardScaler(**text_clf.get_params()["steps"][1][1].get_params())),
                ("clf", SGDClassifier(**text_clf.get_params()["steps"][2][1].get_params()))
            ]
        )
        text_clf["clf"].set_params(tol=1e-4, max_iter=10000, verbose=1, n_jobs=8)

    text_clf.fit(X_train, y_train)

    return text_clf


def grid_search(X, y, n, logging_path):
    text_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 4), analyzer="char", sublinear_tf=True, max_features=150_000)),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", SGDClassifier(
                loss="hinge",
                warm_start=True,
                learning_rate="optimal",
                max_iter=10000,
                tol=1e-3,
                # verbose=1,
                n_iter_no_change=5,
                n_jobs=16
            )),
        ]
        , verbose=False
    )

    param_grid = {
        "tfidf__ngram_range": [(1, 4), (1, 5), (1, 6), (1, 7)],
        "tfidf__sublinear_tf": [True],
        "tfidf__use_idf": [True],
        "tfidf__min_df": [1, 10, 100],
        "tfidf__max_df": [1.0],
        "clf__class_weight": [None, "balanced"],
        "clf__loss": ["log_loss"]
    }

    test_grid = {
        "tfidf__ngram_range": [(1, 1)],
        "clf__loss": ["hinge", "log_loss"]
    }

    grid_clf = GridSearchCV(
        estimator=text_clf,
        param_grid=param_grid,
        refit="f1_macro",
        n_jobs=1,
        cv=3,
        verbose=2,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_micro", "f1_macro"]
    )

    grid_clf.fit(X, y)

    print(grid_clf.cv_results_["params"][grid_clf.best_index_])
    df = pd.DataFrame.from_dict(grid_clf.cv_results_)
    df.to_json(os.path.join(logging_path, f"cv_results-{n}.json"))

    return grid_clf.best_estimator_


def main():
    """
    Training  routine for Naive Bayes phoneme->DID model
    """
    random.seed(144333)
    datapath = os.path.join("config")
    tr_path = os.path.join(datapath, "snf_train_train_all.jsonl")
    tr_cv_de_path = os.path.join(datapath, "train_de_enriched.jsonl")
    tr_cv_en_path = os.path.join(datapath, "train_en_enriched.jsonl")
    dev_path = os.path.join(datapath, "snf_train_valid.jsonl")
    dev_cv_de_path = os.path.join(datapath, "valid_de_enriched.jsonl")
    dev_cv_en_path = os.path.join(datapath, "valid_en_enriched.jsonl")
    test_path = os.path.join(datapath, "snf_test_test.jsonl")
    test_cv_de_path = os.path.join(datapath, "test_de_enriched.jsonl")
    test_cv_en_path = os.path.join(datapath, "test_en_enriched.jsonl")
    train_set = load_dataset(tr_path) + load_dataset(tr_cv_de_path) + load_dataset(tr_cv_en_path)
    valid_set = load_dataset(dev_path) + load_dataset(dev_cv_de_path) + load_dataset(dev_cv_en_path)
    test_set = load_dataset(test_path) + load_dataset(test_cv_de_path) + load_dataset(test_cv_en_path)

    # tag = random.randint(1_000_000, 9_999_999)

    current_time = datetime.now()
    tag = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    do_load_model = False
    do_grid_search = False
    remove_blank = True
    # if do_load_model:
    #     tag = "2024-11-04-19-10"

    logging_path = f"logging/phoneme_clf-{tag}"
    fig_logging_path = f"logging/phoneme_clf-{tag}/figures"
    os.makedirs(fig_logging_path, exist_ok=True)

    sample_range = list(range(1, 21))
    # sample_range = list(range(1, 5))

    best_clf = None
    # for train_sample in [1, 5, 10]:
    if not do_load_model and do_grid_search:
        X_train, y_train = concat_per_speaker(train_set, remove_blank, n=1)
        best_clf = grid_search(X_train, y_train, 1, logging_path=logging_path)

    valid_sets_for_samples = {}
    test_sets_for_samples = {}

    for test_sample in sample_range:
        X_valid, y_valid = concat_per_speaker(valid_set, remove_blank, n=test_sample)
        X_test, y_test = concat_per_speaker(test_set, remove_blank, n=test_sample)
        valid_sets_for_samples[test_sample] = (X_valid, y_valid)
        test_sets_for_samples[test_sample] = (X_test, y_test)

    with open(os.path.join(logging_path, "test-data.json"), "wt", encoding="utf-8") as ofile:
        json.dump(test_sets_for_samples, ofile, ensure_ascii=False)
    with open(os.path.join(logging_path, "valid-data.json"), "wt", encoding="utf-8") as ofile:
        json.dump(valid_sets_for_samples, ofile, ensure_ascii=False)

    f1_mat = np.zeros((len(sample_range), len(sample_range)))
    for iidx, train_sample in enumerate(sample_range):
        if not do_load_model:
            text_clf = run_double_loop(train_set, train_sample, remove_blank, best_clf)
            dump(text_clf, os.path.join(logging_path, f"text_clf-{train_sample}.joblib"))
        else:
            text_clf = load(os.path.join(logging_path, f"text_clf-{train_sample}.joblib"))
            text_clf["clf"].set_params(n_jobs=16)
        for jidx, test_sample in enumerate(sample_range):
            X_valid, y_valid = valid_sets_for_samples[test_sample]
            X_test, y_test = test_sets_for_samples[test_sample]

            target_names = [x[0] for x in sorted(class_nr_map.items(), key=lambda x: x[1], reverse=False)]

            start = time.time()
            predicted = text_clf.predict(X_valid)
            valid_results = metrics.classification_report(y_valid, predicted, target_names=target_names,
                                                          output_dict=True)

            fig, ax = plt.subplots(figsize=(15, 15))
            metrics.ConfusionMatrixDisplay.from_predictions(
                y_valid, predicted, display_labels=target_names, normalize="true", xticks_rotation="vertical", ax=ax,
                colorbar=False, cmap="plasma")
            plt.savefig(os.path.join(fig_logging_path, f"conf-valid-{train_sample}-{test_sample}.png"))
            plt.close(fig)

            predicted = text_clf.predict(X_test)
            test_results = metrics.classification_report(y_test, predicted, target_names=target_names, output_dict=True)

            fig, ax = plt.subplots(figsize=(15, 15))
            metrics.ConfusionMatrixDisplay.from_predictions(
                y_test, predicted, display_labels=target_names, normalize="true", xticks_rotation="vertical", ax=ax,
                colorbar=False, cmap="plasma")
            plt.savefig(os.path.join(fig_logging_path, f"conf-test-{train_sample}-{test_sample}.png"))
            plt.close(fig)

            test_f1_score = test_results["macro avg"]["f1-score"]
            f1_mat[iidx, jidx] = test_f1_score
            results = {
                "valid_results": valid_results,
                "test_results": test_results
            }
            with open(os.path.join(logging_path, f"res-{train_sample}-{test_sample}.json"), "wt",
                      encoding="utf-8") as ofile:
                json.dump(results, ofile, ensure_ascii=False)

            end = time.time()
            print(train_sample, test_sample, test_f1_score, end - start)

    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(f1_mat, cmap="coolwarm")

    # Add a color bar to show the scale
    fig.colorbar(cax)

    # Add labels to the axes, if appropriate
    ax.set_xlabel("Nr. Test Concat")
    ax.set_ylabel("Nr. Train Concat")
    ax.set_xticks(np.arange(len(sample_range)))
    ax.set_yticks(np.arange(len(sample_range)))
    ax.set_xticklabels(sample_range)
    ax.set_yticklabels(sample_range)

    for i in range(len(sample_range)):
        for j in range(len(sample_range)):
            text = ax.text(j, i, f"{f1_mat[i, j]:.2f}", ha="center", va="center", color="white")

    # Optionally add title
    plt.title("F1 Score Heatmap")

    plt.savefig(os.path.join(fig_logging_path, f"all_matrix.pdf"))


if __name__ == "__main__":
    main()
