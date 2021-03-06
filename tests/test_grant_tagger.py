import pytest

import pandas as pd
import numpy as np

from nutrition_labels.grant_tagger import GrantTagger


training_data = pd.DataFrame(
    [
        {
            'text_field': 'Genetics grant to help medicine.',
            'text_field_2': 'Genes linked to illnesses.',
            'Label': 0,
            'ID': 4
        },
        {
            'text_field': 'The history of medicine.',
            'text_field_2': 'Books about medicine and genes.',
            'Label': 0,
            'ID': 1
        },
        {
            'text_field': 'Creating software tools to further technology.',
            'text_field_2': 'Coding in Python.',
            'Label': 1,
            'ID': 2
        },
        {
            'text_field': 'Technology tools will be created.',
            'text_field_2': 'Python and other languages.',
            'Label': 1,
            'ID': 0
        },
        {
            'text_field': 'In this grant we hope to create new software',
            'text_field_2': 'Tools will be created.',
            'Label': 1,
            'ID': 3
        },
        {
            'text_field': 'Software will be created.',
            'text_field_2': 'Machine learning tools.',
            'Label': 1,
            'ID': 5
        }
    ]
    )


prediction_cols = ['text_field', 'text_field_2']
label_name = 'Label'
train_data_id = 'ID'

def test_fit_transform():

    grant_tagger = GrantTagger(
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    X_train = training_data['text_field'].tolist()
    y_train = training_data['Label'].tolist()

    grant_tagger.fit(X_train, y_train)
    X_vect = grant_tagger.transform(pd.DataFrame({'Grant texts': X_train}))

    assert X_vect.shape[0] == 6
    assert X_vect.shape == grant_tagger.X_train_vect.shape

def test_split_data():

    grant_tagger = GrantTagger(
        test_size=1/3,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    train_data, test_data, _ = grant_tagger.split_data(training_data, train_data_id)

    (_, y_train, train_ids) = train_data
    (_, y_test, _) = test_data

    assert train_ids == [2, 0, 5, 4]
    assert len(y_train) == 4
    assert len(y_test) == 2

def test_split_relevant_sample_ratio():

    grant_tagger = GrantTagger(
        relevant_sample_ratio=0.25,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    train_data, test_data, _ = grant_tagger.split_data(training_data, train_data_id)

    (_, y_train, _) = train_data
    (_, y_test, _) = test_data

    all_y = y_train + y_test
    assert len(all_y) == 5
    assert len([y for y in all_y if y==0]) == 1

    grant_tagger = GrantTagger(
        relevant_sample_ratio=0.5,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    training_data_cp = training_data.copy()
    training_data_cp['Label'] = [0, 0, 0, 0, 1, 1]
    train_data, test_data, _ = grant_tagger.split_data(training_data_cp, train_data_id)

    (_, y_train, _) = train_data
    (_, y_test, _) = test_data

    assert len(y_train + y_test) == 3

    grant_tagger = GrantTagger(
        relevant_sample_ratio=1,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )
    train_data, test_data, _ = grant_tagger.split_data(training_data_cp, train_data_id)

    (_, y_train, _) = train_data
    (_, y_test, _) = test_data

    assert len(y_train + y_test) == 4

    grant_tagger = GrantTagger(
        relevant_sample_ratio=2,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )
    train_data, test_data, _ = grant_tagger.split_data(training_data_cp, train_data_id)

    (_, y_train, _) = train_data
    (_, y_test, _) = test_data

    assert len(y_train + y_test) == 6

def test_train_test_info():

    grant_tagger = GrantTagger(
        relevant_sample_ratio=1,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )
    train_data, test_data, unseen_data = grant_tagger.split_data(training_data, train_data_id)

    (X_train, y_train, train_ids) = train_data

    grant_tagger.fit(X_train, y_train)
    grant_info = grant_tagger.train_test_info(train_ids, y_train, test_data, unseen_data)

    training_data_truth_dict = dict(zip(training_data.ID, training_data.Label))
    output_truth_dict = {k:v['Truth'] for k, v in grant_info.items()}

    assert output_truth_dict == training_data_truth_dict

def test_apply_threshold():
    y_predict = [0, 0, 0, 1, 1, 1]
    pred_probs = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
        ]
        )
    grant_tagger = GrantTagger(
        threshold=0.7
        )
    y_predict_thresh = grant_tagger.apply_threshold(y_predict, pred_probs)

    assert all([y1==y2 for y1, y2 in zip([0, 0, 0, 1, 1, 0], y_predict_thresh)]) 
