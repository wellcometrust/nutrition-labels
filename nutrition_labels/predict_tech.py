"""
Output the predictions of the test data from various models

This takes a set of models (found from a particular datestamp - "models_date")
and splits the data in "training_data_file" to the same test/train
split that the models were trained on.
Then the test data found from this is merged with the 
grants text in "grants_data_file" and predictions are made using each of the models.

Outputs:

- data/processed/model_test_results/model_test_results_{models_date}.csv
    The predictions for each model on the test data

- data/processed/model_test_results/models_summary_{models_date}.jsonl
    The test metrics for each model

- data/processed/model_test_results/test_data_{models_date}.csv
    The test data separated out from all the training data

Usage: 
python nutrition_labels/predict_tech.py 
    --training_data_file data/processed/training_data/210221/training_data.csv
    --grants_data_file data/raw/wellcome-grants-awarded-2005-2019.csv
    --models_date 210221
"""

import os
import re
import ast
from collections import defaultdict
from argparse import ArgumentParser
import json

import pandas as pd

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.utils import clean_grants_data

def get_model_dirs(models_date):

    """
    Find models from the models folder which have date of models_date
    """

    model_dirs = os.listdir('models')
    model_dirs.remove('.DS_Store')
    model_dirs = [model_dir for model_dir in model_dirs if (
            model_dir[-6:].isnumeric()) and (
                int(model_dir[-6:])==models_date
                )
            ]
    return model_dirs

def find_model_info(model_dirs):
    """
    Get their metrics from the training text file for the models in model_dirs
    """

    def clean_line_text(line_name, line_results):
        line_results = re.sub(line_name, '', line_results)
        return ast.literal_eval(line_results)


    model_info = defaultdict(dict)
    seeds = set()
    training_dirs = set()
    for model_dir in model_dirs:
        with open('models/' + model_dir + "/training_information.txt", "r") as f:
            for line in f.readlines():
                if 'Test scores: ' in line:
                    results_dict = clean_line_text('Test scores: ', line)
                if 'Split random seed' in line:
                    seeds.add(clean_line_text('Split random seed: ', line))
        
        model_info[model_dir].update({
            'f1': results_dict['f1'],
            'precision_score': results_dict['precision_score'],
            'recall_score': results_dict['recall_score']
            })

    return model_info, seeds

def process_grants_data(
        grant_data,
        training_data,
        split_seed,
        text_columns =['Title', 'Grant Programme:Title', 'Description']
        ):
    """
    Clean the grants text data for the data in the test data split
    when the models were trained (using split_seed)
    """
    
    grant_tagger = GrantTagger(relevant_sample_ratio=1, vectorizer_type='bert')
    _, test_data, _ = grant_tagger.split_data(
        training_data['Internal ID'],
        training_data['Relevance code'],
        split_seed=split_seed)

    (x_test, _, _) = test_data

    test_grant_data = grant_data.loc[grant_data['Internal ID'].isin(x_test)]
    # Add the manually tagged relevance code label from the training data
    test_grant_data = pd.merge(
        test_grant_data[['Internal ID'] + text_columns],
        training_data[['Internal ID', 'Relevance code']],
        how = 'left',
        on = ['Internal ID']
        )

    # Process grants data for predicting
    test_grant_data = clean_grants_data(test_grant_data)
    test_grant_data['Grant texts'] = test_grant_data[text_columns].agg(
                '. '.join, axis=1
                ).tolist()

    return test_grant_data[['Internal ID', 'Relevance code', 'Grant texts']]


def label_grants(model_name, grants_text):

    # Loading a trained model and vectorizer to predict on all the grants data:
    grant_tagger_loaded = GrantTagger()
    grant_tagger_loaded.load_model(os.path.join('models', model_name))

    new_grants_vect = grant_tagger_loaded.vectorizer.transform(grants_text)
    predictions = grant_tagger_loaded.predict(new_grants_vect)

    return predictions # yield?

def create_argparser():

    parser = ArgumentParser()
    parser.add_argument(
        '--training_data_file',
        help='Path to the training data csv',
        default='data/processed/training_data/210221/training_data.csv',
        type=str
    )
    parser.add_argument(
        '--grants_data_file',
        help='Path to the grants data csv for prediction text',
        default='data/raw/wellcome-grants-awarded-2005-2019.csv',
        type=str
    )
    parser.add_argument(
        '--models_date',
        help='The date stamp the models were trained on',
        default='210221',
        type=str
    )

    return parser


if __name__ == '__main__':
    
    parser = create_argparser()
    args = parser.parse_args()

    training_data = pd.read_csv(args.training_data_file)
    grant_data = pd.read_csv(args.grants_data_file)

    models_date = int(args.models_date)

    # Get the models and their info from these dates
    model_dirs = get_model_dirs(models_date)
    model_info, seeds = find_model_info(model_dirs)

    # Some checks
    if len(model_dirs) == 0:
        raise ValueError(
            "There were no models found from this date - try another"
            )
    else:
        print(f'{len(model_dirs)} models found from {models_date}')

    if len(seeds) > 1:
        raise ValueError(
            "There is more than one split train/test seed used to train", 
            "these models - Can't reproduce test set"
            )
    else:
        split_seed = list(seeds)[0]

    # Get and clean the test data
    grant_data = process_grants_data(grant_data, training_data, split_seed)
    ids = grant_data['Internal ID']
    labels = grant_data['Relevance code']
    texts = grant_data['Grant texts']

    model_predictions = defaultdict(list)
    for model_name in model_info.keys():
        print(f'Predicting for {model_name}...')
        model_predictions[model_name] = label_grants(model_name, texts)

    model_predictions_df = pd.DataFrame(model_predictions)
    model_predictions_df['Internal ID'] = ids

    model_predictions_df.to_csv(f'data/processed/model_test_results/model_test_results_{models_date}.csv', index=False)

    with open(f'data/processed/model_test_results/models_summary_{models_date}.jsonl', 'w') as json_file:
        r = json.dumps(model_info)
        json_file.write(str(r))

    grant_data.to_csv(f'data/processed/model_test_results/test_data_{models_date}.csv', index=False)


