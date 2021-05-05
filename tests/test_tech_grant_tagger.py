import pytest
import os

import pandas as pd

from nutrition_labels.ensemble_grant_tagger import EnsembleGrantTagger
from nutrition_labels.grant_tagger import GrantTagger

grants_data = pd.DataFrame([
    {'ID': 123, 'Title': 'This is a title about a grant', 'Relevance code': 0},
    {'ID': 456, 'Title': 'Research about cancer cells', 'Relevance code': 0},
    {'ID': 789, 'Title': 'We will build a python based mathematical model with open source code on github', 'Relevance code': 1}])

X_train = grants_data['Title'].tolist()
y_train = grants_data['Relevance code'].tolist()

def test_load_grants_text(tmp_path):

    grants_data_dir = os.path.join(tmp_path, 'test.csv')
    grants_data.to_csv(grants_data_dir)

    model_dir = os.path.join(tmp_path, 'model_name')
    grant_tagger = GrantTagger(vectorizer_type='count', classifier_type='naive_bayes')
    grant_tagger.fit(X_train, y_train)
    grant_tagger.save_model(model_dir)

    tech_grant_model = EnsembleGrantTagger(
        model_dirs=[model_dir],
        num_agree=1,
        grant_text_cols=['Title'],
        grant_id_col='ID')

    loaded_grants_data = tech_grant_model.load_grants_text(grants_data_dir)
    assert len(grants_data)==3

def test_predict(tmp_path):

    grants_data_dir = os.path.join(tmp_path, 'test.csv')
    grants_data.to_csv(grants_data_dir)

    model_dir = os.path.join(tmp_path, 'model_name')
    grant_tagger = GrantTagger(vectorizer_type='count', classifier_type='naive_bayes')
    grant_tagger.fit(X_train, y_train)
    grant_tagger.save_model(model_dir)

    tech_grant_model = EnsembleGrantTagger(
        model_dirs=[model_dir],
        num_agree=1,
        grant_text_cols=['Title'],
        grant_id_col='ID')

    loaded_grants_data = tech_grant_model.load_grants_text(grants_data_dir)
    final_predictions = tech_grant_model.predict(loaded_grants_data)
    assert len(final_predictions)==3
    
