import pytest

import pandas as pd

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

def test_transform():

    grant_tagger = GrantTagger(
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    X_vect, y = grant_tagger.transform(training_data, train_data_id)

    assert X_vect.shape[0] == 6
    assert grant_tagger.X_ids == [4, 1, 2, 0, 3, 5]

def test_split_data():

    grant_tagger = GrantTagger(
        test_size=1/3,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    X_vect, y = grant_tagger.transform(training_data, train_data_id)
    X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)
    assert len(y_train) == 4
    assert len(y_test) == 2

def test_split_relevant_sample_ratio():

    grant_tagger = GrantTagger(
        relevant_sample_ratio=0.25,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    X_vect, y = grant_tagger.transform(training_data, train_data_id)

    X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)
    all_y = y_train + y_test
    assert len(all_y) == 5
    assert len([y for y in all_y if y==0]) == 1

    grant_tagger = GrantTagger(
        relevant_sample_ratio=0.5,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    y = [0, 0, 0, 0, 1, 1]
    
    X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)
    assert len(y_train + y_test) == 3

    grant_tagger = GrantTagger(
        relevant_sample_ratio=1,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )
    X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)
    assert len(y_train + y_test) == 4

    grant_tagger = GrantTagger(
        relevant_sample_ratio=2,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )
    X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)
    assert len(y_train + y_test) == 6

def test_train_test_info():

    grant_tagger = GrantTagger(
        relevant_sample_ratio=1,
        prediction_cols=prediction_cols,
        label_name=label_name,
        )

    X_vect, y = grant_tagger.transform(training_data, train_data_id)
    X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)
    grant_tagger.fit(X_train, y_train)
    grant_info = grant_tagger.train_test_info()

    training_data_truth_dict = dict(zip(training_data.ID, training_data.Label))
    output_truth_dict = {k:v['Truth'] for k, v in grant_info.items()}

    assert output_truth_dict == training_data_truth_dict
