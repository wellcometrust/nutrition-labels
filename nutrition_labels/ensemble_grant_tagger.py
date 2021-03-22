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
from datetime import datetime

import pandas as pd

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.utils import remove_useless_string

class EnsembleGrantTagger():
    def __init__(
        self,
        model_dirs,
        num_agree=3,
        grant_text_cols = ['Title', 'Grant Programme:Title', 'Description'],
        grant_id_col = 'Internal ID'):
        self.model_dirs = model_dirs
        self.num_agree = num_agree
        self.grant_text_cols = grant_text_cols
        self.grant_id_col = grant_id_col


    def clean_grants_data(self, previous_grant_data):
        """
        Clean grant descriptions of html
        Merge the grant descriptions columns into one
        """
        grant_data = previous_grant_data.copy()
        grant_data.fillna('', inplace=True)
        grant_data[self.grant_text_cols] = grant_data[self.grant_text_cols].applymap(
            remove_useless_string
            )

        grant_data['Grant texts'] = grant_data[self.grant_text_cols].agg(
                '. '.join, axis=1
                ).tolist()
        return grant_data

    def load_grants_text(self, grants_data_path):
        """
        Field names need an update here!
        """
        grant_data = pd.read_csv(grants_data_path)
        grant_data = self.clean_grants_data(grant_data)
        grants_text = grant_data['Grant texts'].tolist()
        self.grants_ids = grant_data[self.grant_id_col].tolist()

        return grants_text

    def load_model(self, model_path):

        # Loading a trained model and vectorizer to predict on all the grants data:
        grant_tagger_loaded = GrantTagger()
        grant_tagger_loaded.load_model(model_path)

        return grant_tagger_loaded

    def predict_tags(self, grant_tagger_loaded, grants_text):

        new_grants_vect = grant_tagger_loaded.vectorizer.transform(grants_text)
        predictions = grant_tagger_loaded.predict(new_grants_vect)

        # # Do something if grants_text is ' . . ' or 'Not available' or len<threshold?
        # grant_data = grant_data[grant_data['Description'] != 'Not available']
        # grant_data.dropna(subset=['Description'], inplace=True)

        return predictions


    def predict(self, grants_text):
        """
        Predict whether grant texts (list) are tech grants or not 
        using an agreement of num_agree
        of the models in model_dirs
        """

        model_predictions = {}
        for model_dir in self.model_dirs:
            model_name = os.path.basename(model_dir)
            print(f'Predicting for {model_name}...')
            grant_tagger_loaded = self.load_model(model_dir)
            model_predictions[f'{model_name} predictions'] = self.predict_tags(
                grant_tagger_loaded,
                grants_text
                )

        model_predictions_df = pd.DataFrame(model_predictions)
        prediction_sums = model_predictions_df.sum(axis=1) 
 
        self.final_predictions = (prediction_sums >= self.num_agree).astype(int).tolist()
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
        default='configs/ensemble/2021.02.20.ini'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    datestamp = datetime.now().date().strftime('%y%m%d')

    grants_data_path = config["prediction_data"]["grants_data_path"]
    input_file_name = os.path.basename(grants_data_path).split('.')[0]
    output_path = f'data/processed/ensemble/{datestamp}/{input_file_name}_tagged.csv'

    tech_grant_model = EnsembleGrantTagger(
        model_dirs=config["ensemble_model"]["model_dirs"].split(','), # ['models/count_naive_bayes_210218']
        num_agree=config.getint("ensemble_model", "num_agree"),
        grant_text_cols=config["prediction_data"]["grant_text_cols"].split(','),
        grant_id_col=config["prediction_data"]["grant_id_col"])

    grants_text = tech_grant_model.load_grants_text(grants_data_path)
    final_predictions = tech_grant_model.predict(grants_text)
    tech_grant_model.output_tagged_grants(output_path)

    print(f'{sum(final_predictions)} tech grants predicted in {len(final_predictions)} grants')
