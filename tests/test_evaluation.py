import pytest
import os
import json

import pandas as pd

from nutrition_labels.evaluate import get_training_data, EvaluateGrantTagger

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

expected_training_data = pd.DataFrame([
        {'Grant ID': 'id_A', 'Truth': 1, 'Test/train': 'Train'},
        {'Grant ID': 'id_B', 'Truth': 0, 'Test/train': 'Test'},
        {'Grant ID': 'id_C', 'Truth': 1, 'Test/train': 'Test'},
        {'Grant ID': 'id_D', 'Truth': 0, 'Test/train': 'Test'}
        ])

def test_get_training_data(tmp_path):

    with open(os.path.join(tmp_path, "training_information.json"), "a") as f:
        for line in model_scores:
            f.write(json.dumps(line))
            f.write('\n')
    training_data = get_training_data(tmp_path, grant_id_col='Grant ID')

    assert training_data.equals(expected_training_data)

def test_find_datasets_included():

    eval_grant_tagger = EvaluateGrantTagger(
            model_dirs=['model_dir'],
            epmc_file_dir='epmc_file_dir.csv',
            rf_file_dir='rf_file_dir.csv'
            )
    eval_grant_tagger.training_data = expected_training_data
    datasets_included = eval_grant_tagger.find_datasets_included()

    assert set(datasets_included) == set(['test', 'EPMC', 'RF'])

def test_return_dataset():

    eval_grant_tagger = EvaluateGrantTagger(
            model_dirs=['model_dir']
            )
    eval_grant_tagger.training_data = expected_training_data
    eval_grant_tagger.training_label_name = 'Truth'

    dataset_info = eval_grant_tagger.return_dataset("test")
    
    assert len(dataset_info["dataset"]) == 3
    assert 'id_A' not in dataset_info["dataset"]['Grant ID']
    assert dataset_info["label_name"] == 'Truth'
