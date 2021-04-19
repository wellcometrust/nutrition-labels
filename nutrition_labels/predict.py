"""
For a dataset load a model (or ensemble of models) and predict whether the grants
are tech grants or not.

Run as:
python nutrition_labels/predict.py --config_path configs/predict/2021.04.02.ini

"""

from argparse import ArgumentParser
import configparser
import os

import pandas as pd

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.ensemble_grant_tagger import EnsembleGrantTagger

def output_tagged_grants(output_path, y_pred, grant_ids):

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    pd.DataFrame(
        {
        'Tech grant prediction': y_pred,
        'Grant ID': grant_ids
        }
        ).to_csv(output_path, index=False)

def predict_grants(config):

    grants_data_path = config["prediction_data"]["grants_data_path"] 
    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]
    input_file_name = os.path.basename(grants_data_path).split('.')[0]
    output_path = f'data/processed/ensemble/{config_version}/{input_file_name}_tagged.csv'

    try:
        pred_prob_thresh = config.getfloat("model_parameters", "pred_prob_thresh")
    except:
        pred_prob_thresh = None

    model_dirs = config["model_parameters"]["model_dirs"].split(',')
    grant_text_cols = config["prediction_data"]["grant_text_cols"].split(',')
    grant_id_col = config["prediction_data"]["grant_id_col"]

    grants_data = pd.read_csv(grants_data_path)
    grant_ids = grants_data[grant_id_col].tolist()

    if len(model_dirs) == 1:
        model_dir = model_dirs[0]
        grant_tagger = GrantTagger(
            threshold=pred_prob_thresh,
            prediction_cols=grant_text_cols,
            )
        grant_tagger.load_model(model_dir)
        X_vec = grant_tagger.transform(grants_data)
        y_pred = grant_tagger.predict(X_vec)
    else:
        tech_grant_model = EnsembleGrantTagger(
            model_dirs=model_dirs,
            num_agree=config.getint("model_parameters", "num_agree"),
            grant_text_cols=grant_text_cols,
            grant_id_col=grant_id_col,
            threshold=pred_prob_thresh
            )
        grants_text = tech_grant_model.load_grants_text(grants_data_path)
        y_pred = tech_grant_model.predict(grants_text)

    output_tagged_grants(output_path, y_pred, grant_ids)

    return grant_ids, y_pred

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument(
        '--config_path',
        help='Path to config file',
        default='configs/predict/2021.04.02.ini'
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    grant_ids, y_pred = predict_grants(config)

    print(f'{sum(y_pred)} tech grants predicted in {len(y_pred)} grants')
