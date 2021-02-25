import pytest
import tempfile
import os

import pandas as pd

from nutrition_labels.tech_grant_tagger import TechGrantModel


grants_data = pd.DataFrame([
    {'ID': 123, 'Title': 'This is a title about a grant'},
    {'ID': 456, 'Title': 'Research about cancer cells'},
    {'ID': 789, 'Title': 'We will build a python based mathematical model with open source code on github'}])

def test_load_grants_text():

    with tempfile.TemporaryDirectory() as tmp_dir:
        grants_data_dir = os.path.join(tmp_dir, 'test.csv')
        grants_data.to_csv(grants_data_dir)

        tech_grant_model = TechGrantModel(
            model_dirs=['models/count_naive_bayes_210221'],
            num_agree=1,
            grant_text_cols=['Title'],
            grant_id_col='ID')

        loaded_grants_data = tech_grant_model.load_grants_text(grants_data_dir)
        assert len(grants_data)==3

def test_predict():

    with tempfile.TemporaryDirectory() as tmp_dir:
        grants_data_dir = os.path.join(tmp_dir, 'test.csv')
        grants_data.to_csv(grants_data_dir)

        tech_grant_model = TechGrantModel(
            model_dirs=['models/count_naive_bayes_210221'],
            num_agree=1,
            grant_text_cols=['Title'],
            grant_id_col='ID')

        loaded_grants_data = tech_grant_model.load_grants_text(grants_data_dir)
        final_predictions = tech_grant_model.predict(loaded_grants_data)
        assert len(final_predictions)==3
    
