import pytest
import os
import json

import pandas as pd

from nutrition_labels.grant_tagger_evaluation import get_model_test_scores, get_evaluation_data_scores
from nutrition_labels.grant_tagger import GrantTagger

model_scores = [
    {
        "model_1": {
            "id_A": {"Truth": 1, "Prediction": 1, "Test/train": "Train"},
            "id_B": { "Truth": 0, "Prediction": 1, "Test/train": "Test"},
            "id_C": {"Truth": 1, "Prediction": 0, "Test/train": "Test"},
            "id_D": {"Truth": 0, "Prediction": 1, "Test/train": "Test"}
            }
    },
    {
        "model_2": {
            "id_A": {"Truth": 1, "Prediction": 1, "Test/train": "Train"},
            "id_B": { "Truth": 0, "Prediction": 0, "Test/train": "Test"},
            "id_C": {"Truth": 1, "Prediction": 1, "Test/train": "Test"},
            "id_D": {"Truth": 0, "Prediction": 0, "Test/train": "Test"}
            }
    },
    ]

def test_get_model_test_scores(tmp_path):

    with open(os.path.join(tmp_path, "training_information.json"), "a") as f:
        for line in model_scores:
            f.write(json.dumps(line))
            f.write('\n')
    all_models_info = get_model_test_scores(tmp_path)

    assert len(all_models_info) == 2
    assert all_models_info['model_1']['f1'] == 0
    assert all_models_info['model_2']['f1'] == 1


def test_get_evaluation_data_scores(tmp_path):

    training_data = pd.DataFrame([
        {'Internal ID': "id_A", 'Relevance code': 1, 'Title': 'Software technology'},
        {'Internal ID': "id_B", 'Relevance code': 0, 'Title': 'A funding application'}])

    X_train = training_data['Title'].tolist()
    y_train = training_data['Relevance code'].tolist()

    model_names = ['count', 'tfidf']
    for vectorizer in model_names:
        grant_tagger = GrantTagger(
            vectorizer_type=vectorizer,
            classifier_type='naive_bayes'
            )
        grant_tagger.fit(X_train, y_train)
        grant_tagger.save_model(os.path.join(tmp_path, vectorizer))

    epmc_results, _ = get_evaluation_data_scores(
        tmp_path,
        model_names,
        prediction_cols=['Title'],
        epmc_evaluation_data=training_data,
        rf_evaluation_data=training_data
        )
    assert set(epmc_results.keys()) == set(model_names)
    assert epmc_results[model_names[0]] <= 1
    