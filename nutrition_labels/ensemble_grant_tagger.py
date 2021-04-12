"""

Get predictions using an ensemble of models.

The data to predict on and the number of models to agree on, 
and the names of all the models to include should be stored in 
a config file, e.g. configs/ensemble/2021.02.19.ini

The output will be a csv of IDs and tech predictions.

python ensemble_grant_tagger.py --config_path configs/ensemble/2021.02.19.ini

"""
from argparse import ArgumentParser
import configparser
import os

import pandas as pd
from sklearn.pipeline import Pipeline

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.utils import clean_string

from wellcomeml.ml import WellcomeVotingClassifier

class EnsembleGrantTagger():
    def __init__(
        self,
        model_dirs,
        num_agree=3,
        grant_text_cols = ['Title', 'Grant Programme:Title', 'Description'],
        grant_id_col = 'Internal ID',
        pred_prob_threshold = None):
        self.model_dirs = model_dirs
        self.num_agree = num_agree
        self.grant_text_cols = grant_text_cols
        self.grant_id_col = grant_id_col
        self.pred_prob_threshold = pred_prob_threshold

    def load_grants_text(self, grants_data_path):
        """
        Load and clean grant descriptions for predictions.
        Do this once here rather that for each model in the ensemble.
        """
        grant_data = pd.read_csv(grants_data_path)

        grant_tagger = GrantTagger(prediction_cols=self.grant_text_cols)
        grant_data = grant_tagger.process_grant_text(grant_data)

        grants_text = grant_data['Grant texts'].tolist()
        self.grants_ids = grant_data[self.grant_id_col].tolist()

        return grants_text

    def load_model(self, model_path):

        # Loading a trained model and vectorizer to predict on all the grants data:
        grant_tagger_loaded = GrantTagger(pred_prob_threshold=self.pred_prob_threshold)
        grant_tagger_loaded.load_model(model_path)

        return grant_tagger_loaded

    def predict(self, grants_text):
        """
        Predict whether grant texts (list) are tech grants or not 
        using an agreement of num_agree of the models in model_dirs
        """

        ensemble_pipelines = []
        for model_dir in self.model_dirs:
            grant_tagger_loaded = self.load_model(model_dir)
            ensemble_pipelines.append(
                Pipeline([('vect', grant_tagger_loaded), ('est', grant_tagger_loaded)])
                )

        voting_classifier = WellcomeVotingClassifier(
            estimators=ensemble_pipelines,
            voting="hard",
            num_agree=self.num_agree
        )

        self.final_predictions = voting_classifier.predict(
            pd.DataFrame({"Grant texts": grants_text})
            )

        return self.final_predictions

    def output_tagged_grants(self, output_path):

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        pd.DataFrame(
            {
            'Tech grant prediction': self.final_predictions,
            'Grant ID': self.grants_ids
            }
            ).to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--config_path',
        help='Path to config file',
        default='configs/ensemble/2021.04.01.ini'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]

    grants_data_path = config["prediction_data"]["grants_data_path"]
    input_file_name = os.path.basename(grants_data_path).split('.')[0]
    output_path = f'data/processed/ensemble/{config_version}/{input_file_name}_tagged.csv'

    try:
        pred_prob_thresh = config.getfloat("ensemble_model", "pred_prob_thresh")
    except:
        pred_prob_thresh = None

    tech_grant_model = EnsembleGrantTagger(
        model_dirs=config["ensemble_model"]["model_dirs"].split(','),
        num_agree=config.getint("ensemble_model", "num_agree"),
        grant_text_cols=config["prediction_data"]["grant_text_cols"].split(','),
        grant_id_col=config["prediction_data"]["grant_id_col"],
        pred_prob_threshold=pred_prob_thresh
        )

    grants_text = tech_grant_model.load_grants_text(grants_data_path)
    final_predictions = tech_grant_model.predict(grants_text)
    tech_grant_model.output_tagged_grants(output_path)

    print(f'{sum(final_predictions)} tech grants predicted in {len(final_predictions)} grants')
