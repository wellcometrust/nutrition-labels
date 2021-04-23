"""
For a single or ensemble tech model:

- Load model(s)
- Predict on test data
- Predict on unseen data
- Predict on EPMC and RF data (if given)
- Output all evaluation results

Run as:
python nutrition_labels/evaluate.py --config_path configs/evaluation/2021.04.02.ini

"""

from argparse import ArgumentParser
import configparser
import os
import json

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.ensemble_grant_tagger import EnsembleGrantTagger
from nutrition_labels.utils import pretty_confusion_matrix

def merge_grants(eval_data, grant_data, eval_id_col, grant_id_col, label_name):
    eval_data = pd.merge(
            eval_data,
            grant_data,
            how="left",
            left_on=eval_id_col,
            right_on=grant_id_col
        )
    eval_data[label_name] = eval_data[label_name].astype(int)
    return eval_data

def get_training_data(model_dir, grant_id_col):
    
    training_info_file = os.path.join(model_dir, 'training_information.json')
    with open(training_info_file, 'r') as file:
        for line in file:
            model_data = json.loads(line)
            model_name = list(model_data.keys())[0]
            # The ground truth will be the same for every model, so no need to read every line
            break
    training_data = [(grant_id, m['Truth'], m['Test/train']) for grant_id, m in model_data[model_name].items()]
    training_data = pd.DataFrame(training_data, columns = [grant_id_col, 'Truth', 'Test/train'])

    return training_data

class EvaluateGrantTagger():

    def __init__(
        self,
        model_dirs,
        grant_text_cols=['Title', 'Synopsis'],
        grant_id_col='Reference',
        pred_prob_thresh=None,
        epmc_file_dir=None,
        rf_file_dir=None,
        eval_id_col=None,
        eval_label_name=None,
        average = "binary"
        ):
        self.model_dirs = model_dirs
        self.grant_text_cols = grant_text_cols
        self.grant_id_col = grant_id_col
        self.pred_prob_thresh = pred_prob_thresh
        self.epmc_file_dir = epmc_file_dir
        self.rf_file_dir = rf_file_dir
        self.eval_id_col = eval_id_col
        self.eval_label_name = eval_label_name
        self.average = average

    def load_grant_data(self, grants_data_path):

        grant_data = pd.read_csv(grants_data_path)
        self.grant_data = grant_data.drop_duplicates(subset=self.grant_id_col)[self.grant_text_cols + [self.grant_id_col]] # Don't need it all

    def load_training_data(self, training_label_name='Truth'):
        # Training data
        model_dir = os.path.dirname(self.model_dirs[0]) # It's the same directory name for all models
        training_data = get_training_data(model_dir, self.grant_id_col)
        self.training_label_name = training_label_name
        self.training_data = merge_grants(training_data, self.grant_data, self.grant_id_col, self.grant_id_col, self.training_label_name)

    def find_datasets_included(self):
        # Make a list of the datasets included in this evaluation
        datasets_included = []
        if sum(self.training_data['Test/train'] == 'Test') != 0:
            datasets_included.append('test')
        if sum(self.training_data['Test/train'] == 'Not used') != 0:
            datasets_included.append('not seen')
        if self.epmc_file_dir:
            datasets_included.append('EPMC')
        if self.rf_file_dir:
            datasets_included.append('RF')

        return datasets_included

    def create_test_dataset(self):
        return self.training_data.loc[self.training_data['Test/train'] == 'Test']

    def create_unseen_dataset(self):
        return self.training_data.loc[self.training_data['Test/train'] == 'Not used']

    def create_epmc_dataset(self):
        epmc_evaluation_data = pd.read_csv(self.epmc_file_dir)
        epmc_evaluation_data = merge_grants(epmc_evaluation_data, self.grant_data, self.eval_id_col, self.grant_id_col, self.eval_label_name)
        return epmc_evaluation_data

    def create_rf_dataset(self):
        rf_evaluation_data = pd.read_csv(self.rf_file_dir)
        rf_evaluation_data = merge_grants(rf_evaluation_data, self.grant_data, self.eval_id_col, self.grant_id_col, self.eval_label_name)
        return rf_evaluation_data

    def return_dataset(self, dataset_id):
        if dataset_id == "test":
            return {
                "dataset": self.create_test_dataset(),
                "label_name": self.training_label_name
                }
        elif dataset_id == "not seen":
            return {
                "dataset": self.create_unseen_dataset(),
                "label_name": self.training_label_name
                }
        elif dataset_id == "EPMC":
            return {
                "dataset": self.create_epmc_dataset(),
                "label_name": self.eval_label_name
                }
        elif dataset_id == "RF":
            return {
                "dataset": self.create_rf_dataset(),
                "label_name": self.eval_label_name
                }
        else:
            print("No dataset for this name")

    def evaluate_single_model(self, model_dir, datasets_included):
        
        grant_tagger = GrantTagger(
            threshold=self.pred_prob_thresh,
            prediction_cols=self.grant_text_cols,
            )
        grant_tagger.load_model(model_dir)

        dataset_scores = {}
        for dataset_name in datasets_included:
            dataset_info = self.return_dataset(dataset_name)
            
            print(f"Evaluating models on {dataset_name} dataset ...")

            X_vec = grant_tagger.transform(dataset_info['dataset'])
            y = dataset_info['dataset'][dataset_info['label_name']].tolist()
            scores = grant_tagger.evaluate(X_vec, y, extra_scores=True, average=self.average)

            dataset_scores[dataset_name] = scores

        return dataset_scores

    def evaluate_ensemble_model(self, model_dirs, num_agree, datasets_included):
        
        ensemble_grant_tagger = EnsembleGrantTagger(
                model_dirs=model_dirs,
                num_agree=num_agree,
                grant_text_cols=self.grant_text_cols,
                grant_id_col=self.grant_id_col,
                threshold=self.pred_prob_thresh
                )
        dataset_scores = {}
        for dataset_name in datasets_included:
            dataset_info = self.return_dataset(dataset_name)
            print(f"Evaluating models on {dataset_name} dataset ...")

            grant_tagger = GrantTagger(prediction_cols=self.grant_text_cols)
            grant_data = grant_tagger.process_grant_text(dataset_info['dataset'])
            grants_text = dataset_info['dataset']['Grant texts'].tolist()
            y_predict = ensemble_grant_tagger.predict(grants_text)
            y = dataset_info['dataset'][dataset_info['label_name']].tolist()
            scores = {
                "size": len(y),
                "accuracy": accuracy_score(y, y_predict),
                "f1": f1_score(y, y_predict, average=self.average),
                "precision_score": precision_score(
                    y, y_predict, zero_division=0, average=self.average
                ),
                "recall_score": recall_score(
                    y, y_predict, zero_division=0, average=self.average
                ),
            }
            scores["Classification report"] = classification_report(y, y_predict)
            scores["Confusion matrix"] = pretty_confusion_matrix(y, y_predict)

            dataset_scores[dataset_name] = scores

        return dataset_scores

def output_scores(dataset_scores, output_path):
    with open(output_path, "w") as output_file:
        for dataset_name, scores in dataset_scores.items():
            output_file.write("\n" + dataset_name + ":\n")
            for score_type, score in scores.items():
                output_file.write("\n" + score_type + ":\n" + str(score))

def evaluate_models(config):

    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]
    model_dirs = config["ensemble_model"]["model_dirs"].split(',')

    try:
        pred_prob_thresh = config.getfloat("ensemble_model", "pred_prob_thresh")
    except:
        pred_prob_thresh = None

    try:
        epmc_file_dir = config.get("prediction_data", "epmc_file_dir")
    except:
        epmc_file_dir = None

    try:
        rf_file_dir = config.get("prediction_data", "rf_file_dir")
    except:
        rf_file_dir = None

    try:
        eval_id_col = config.get("prediction_data", "eval_id_col")
        eval_label_name = config.get("prediction_data", "eval_label_name")
    except:
        eval_id_col = None
        eval_label_name = None

    # Grant data parameters
    grants_data_path = config["prediction_data"]["grants_data_path"]
    grant_text_cols = config["prediction_data"]["grant_text_cols"].split(',')
    grant_id_col = config["prediction_data"]["grant_id_col"]

    eval_grant_tagger = EvaluateGrantTagger(
        model_dirs=model_dirs,
        grant_text_cols=grant_text_cols,
        grant_id_col=grant_id_col,
        pred_prob_thresh=pred_prob_thresh,
        epmc_file_dir=epmc_file_dir,
        rf_file_dir=rf_file_dir,
        eval_id_col=eval_id_col,
        eval_label_name=eval_label_name,
        average='binary'
        )

    eval_grant_tagger.load_grant_data(grants_data_path)
    training_label_name = 'Truth'
    eval_grant_tagger.load_training_data(training_label_name)
    datasets_included = eval_grant_tagger.find_datasets_included()

    # Evaluate and output
    
    output_path = f'data/processed/evaluation/{config_version}/evaluation_results.txt'

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    if len(model_dirs) == 1:
        dataset_scores = eval_grant_tagger.evaluate_single_model(model_dirs[0], datasets_included)
        output_scores(dataset_scores, output_path)
    else:
        num_agree = config.getint("ensemble_model", "num_agree")
        dataset_scores = eval_grant_tagger.evaluate_ensemble_model(model_dirs, num_agree, datasets_included)
        output_scores(dataset_scores, output_path)

    return dataset_scores

def parse_arguments(parser):

    parser.add_argument(
        '--config_path',
        help='Path to config file',
        default='configs/evaluation/2021.04.02.ini'
    )

    return parser.parse_args()

if __name__ == '__main__':
    
    parser = ArgumentParser()
    args = parse_arguments(parser)

    config = configparser.ConfigParser()
    config.read(args.config_path)

    dataset_scores = evaluate_models(config)
