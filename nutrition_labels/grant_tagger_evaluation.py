"""
Script to evaluate trained models on EPMC and RF evaluation datasets. 
These datasets just contain grants linked to tech outputs, so we only output
accuracy, which in this case is the same as recall.
This script also combines these with the test metrics which are outputed when the model was trained.
"""

import json
import ast
import os
from argparse import ArgumentParser
import configparser

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.utils import clean_string


def merge_eval_grants(evaluation_data, grants_data, prediction_cols, grants_text_data_file_id):
    """
    Merge evaluation data with text to predict.
    Use same text columns as the model was trained on.
    """

    evaluation_data = pd.merge(
        evaluation_data,
        grants_data.drop_duplicates(subset=grants_text_data_file_id)[
            prediction_cols + [grants_text_data_file_id]
        ],
        how="left",
        left_on="Internal ID",
        right_on=grants_text_data_file_id,
    )
    evaluation_data['Relevance code'] = evaluation_data['Relevance code'].astype(int)
    
    return evaluation_data

def evaluate_data(evaluation_data, grant_tagger):
    """
    Predict evaluation data using grant_tagger and return the accuracy score.
    """
    X_vect_eval = grant_tagger.transform(evaluation_data)
    y_eval = evaluation_data["Relevance code"].tolist()
    eval_scores = grant_tagger.evaluate(X_vect_eval, y_eval)

    return eval_scores['accuracy']

def get_evaluation_data_scores(
        model_date_dir,
        model_names,
        prediction_cols,
        epmc_evaluation_data,
        rf_evaluation_data
        ):
    """
    Load the models and predict the EPMC and RF evaluation data using each of them
    """

    epmc_results = {}
    rf_results = {}
    for model_name in model_names:
        grant_tagger = GrantTagger(prediction_cols=prediction_cols)
        grant_tagger.load_model(os.path.join(model_date_dir, model_name))
        epmc_results[model_name] = evaluate_data(epmc_evaluation_data, grant_tagger)
        rf_results[model_name] = evaluate_data(rf_evaluation_data, grant_tagger)

    return epmc_results, rf_results

def get_model_test_scores(model_date_dir):
    """
    Get the test scores for each model in model_date_dir using the truth/predictions
    outputted in the models' training information file (no need to predict again).
    """

    training_info_file = os.path.join(model_date_dir, 'training_information.json')
    all_models_info = {}
    with open(training_info_file, 'r') as file:
        for line in file:
            model_data = json.loads(line)
            model_name = list(model_data.keys())[0]
            all_models_info[model_name] = model_data[model_name]
            
            test_results = pd.DataFrame([m for m in model_data[model_name].values() if m['Test/train']=='Test'])
            y_true = test_results['Truth']
            y_pred = test_results['Prediction']
            all_models_info[model_name] = {
                'f1': f1_score(y_true, y_pred),
                'precision_score': precision_score(y_true, y_pred),
                'recall_score': recall_score(y_true, y_pred)
                }

    return all_models_info

def get_model_scores_df(config, args):
    """
    Import data and get scores for the RF and EPMC evaluation data, and the test scores
    for each model. Output in a nice format.
    """

    prediction_cols = ast.literal_eval(config["data"]["prediction_cols"])
    grants_text_data_file = config["data"]["grants_text_data_file"]
    grants_text_data_file_id = config["data"]["grants_text_data_file_id"]

    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]
    model_date_dir = os.path.join('models', config_version)

    model_names = os.listdir(f'{model_date_dir}')
    model_names.remove('training_information.json')
    model_names.remove('.DS_Store')
    try:
        model_names.remove('test_data_formatted.csv')
    except:
        print('test_data_formatted.csv not created yet')

    # Get EPMC and RF evaluation data

    grants_data = pd.read_csv(grants_text_data_file)
    epmc_evaluation_data = pd.read_csv(args.epmc_file_dir)
    rf_evaluation_data = pd.read_csv(args.rf_file_dir)

    epmc_evaluation_data = merge_eval_grants(
        epmc_evaluation_data,
        grants_data,
        prediction_cols,
        grants_text_data_file_id
        )
    rf_evaluation_data = merge_eval_grants(
        rf_evaluation_data,
        grants_data,
        prediction_cols,
        grants_text_data_file_id
        )

    # Get model scores

    epmc_results, rf_results = get_evaluation_data_scores(
        model_date_dir,
        model_names,
        prediction_cols,
        epmc_evaluation_data,
        rf_evaluation_data
        )

    all_models_info = get_model_test_scores(model_date_dir)

    for k, v in epmc_results.items():
        all_models_info[k].update({'EPMC accuracy': v})
    for k, v in rf_results.items():
        all_models_info[k].update({'RF accuracy': v})

    # Save nicely formatted outputß

    scores_df = pd.DataFrame(all_models_info).T.round(3)
    scores_df['Vectorizer'] = [x.split('_')[0] for x in all_models_info.keys()]
    scores_df['Classifier'] = ['_'.join(x.split('_')[1:-1]) for x in all_models_info.keys()]
    scores_df['Date'] = [x.split('_')[-1] for x in all_models_info.keys()]

    column_order = [
        'Date',
        'Vectorizer',
        'Classifier',
        'f1',
        'precision_score',
        'recall_score',
        'EPMC accuracy',
        'RF accuracy'
        ]
    scores_df = scores_df[column_order].reset_index(drop=True)
    
    scores_df.to_csv(os.path.join(model_date_dir, 'test_data_formatted.csv'))

    return scores_df

if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser.add_argument(
        "--model_config",
        help="The path to config file used to train the models",
        default='configs/train_model/2021.03.31.ini',
    )
    parser.add_argument(
        "--epmc_file_dir",
        help="The path to the EPMC evaluation data",
        default='data/processed/training_data/210329epmc/training_data.csv',
    )
    parser.add_argument(
        "--rf_file_dir",
        help="The path to the RF evaluation data",
        default='data/processed/training_data/210329rf/training_data.csv',
    )

    args = parser.parse_args()

    # Get grant data information used in training via the training config file

    config = configparser.ConfigParser()
    config.read(args.model_config)

    scores_df = get_model_scores_df(config, args)
