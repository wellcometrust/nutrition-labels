"""

Ensemble model from 201118
Hardcoded:
- The names of the 4 models included in this
- That we are using agreement of 3 out of 4 models.

By hardcoding it this way, the ensemble model stays 
static for production and doesn't do any recalculations
of e.g. which of the 12 models are best to use in the ensemble,
or how many models should agree

python tech_grant_tagger.py --input_path grants.csv --output_path tagged_grants.csv 
    --models_path models/ensemble_201118/

"""
from argparse import ArgumentParser
import os

import pandas as pd

from grant_tagger import GrantTagger
from nutrition_labels.useful_functions import remove_useless_string

class TechGrantModel():
    def __init__(
        self,
        models_path,
        input_path,
        output_path,
        num_agree=3,
        grant_text_cols = ['Title', 'Grant Programme:Title', 'Description']):
        self.models_path = models_path
        self.input_path = input_path
        self.output_path = output_path
        self.num_agree = num_agree
        self.grant_text_cols = grant_text_cols

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

    def load_grants_text(self):
        """
        Field names need an update here!
        """
        grant_data = pd.read_csv(self.input_path)
        grant_data = self.clean_grants_data(grant_data)
        grants_text = grant_data['Grant texts'].tolist()

        return grants_text

    def predict(self, grants_text):
        """
        Predict whether grant texts (list) are tech grants or not 
        using an agreement of num_agree
        of the models in models_path
        """

        models = os.listdir(self.models_path)
        models.remove('.DS_Store')

        model_predictions = {}
        for model_name in models:
            grant_tagger_loaded = self.load_model(os.path.join(self.models_path, model_name))
            model_predictions[f'{model_name} predictions'] = self.predict_tags(
                grant_tagger_loaded,
                grants_text
                )

        model_predictions_df = pd.DataFrame(model_predictions)
        prediction_sums = model_predictions_df.sum(axis=1) 
 
        self.final_predictions = (prediction_sums >= self.num_agree).astype(int).tolist()
        return self.final_predictions

    def output_tagged_grants(self):

        pd.DataFrame(
            self.final_predictions,
            columns=['Tech grant prediction']
            ).to_csv(self.output_path, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--models_path',
        help='Path to folder of models included in the ensemble',
        default='models/ensemble_210129_models/'
    )
    parser.add_argument(
        '--input_path',
        help='Path to file with grants to predict',
        default='data/raw/wellcome-grants-awarded-2005-2019_test_sample.csv'
    )
    parser.add_argument(
        '--output_path',
        help='Path to output',
        default='data/processed/wellcome-grants-awarded-2005-2019_test_sample_tagged.csv'
    )
    parser.add_argument(
        '--num_agree',
        help='Number of models in models_path which need to agree to tag as tech',
        default=3,
        type=int
    )
    parser.add_argument(
        '--grant_text_cols',
        help='A list of the columns in input_path which you want to merge and predict on',
        default=['Title', 'Grant Programme:Title', 'Description'],
        type=list
    )
    args = parser.parse_args()

    tech_grant_model = TechGrantModel(
        models_path=args.models_path,
        input_path=args.input_path,
        output_path=args.output_path,
        num_agree=args.num_agree,
        grant_text_cols=args.grant_text_cols)

    grants_text = tech_grant_model.load_grants_text()
    final_predictions = tech_grant_model.predict(grants_text)
    tech_grant_model.output_tagged_grants()

