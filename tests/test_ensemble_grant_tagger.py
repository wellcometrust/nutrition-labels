import pytest
import os

import pandas as pd

from nutrition_labels.ensemble_grant_tagger import EnsembleGrantTagger
from nutrition_labels.grant_tagger import GrantTagger

# Designed to give varying results for count and tfidf
text_data = pd.DataFrame(
            [
                {'id':0, 'text_1': 'dog dog', 'text_2': '', 'label': 1},
                {'id':1, 'text_1': 'dog dog', 'text_2': '', 'label': 1},
                {'id':2, 'text_1': 'dog skate', 'text_2': '', 'label': 0},
                {'id':3, 'text_1': 'cat shoe', 'text_2': '', 'label': 0},
                {'id':4, 'text_1': 'dog dog skate cat shoe sandal', 'text_2': '', 'label': 0},
                {'id':5, 'text_1': 'sandal', 'text_2': '', 'label': 0},
                {'id':6, 'text_1': None, 'text_2': None, 'label': 0}
            ]
        )

def test_load_grants_text(tmp_path):

    grants_data_path = os.path.join(tmp_path, 'text_data.csv')
    text_data.to_csv(grants_data_path)
    ensemble_gt = EnsembleGrantTagger(
        model_dirs = [],
        grant_text_cols = ['text_1', 'text_2'],
        grant_id_col = 'id',
        )
    grants_text = ensemble_gt.load_grants_text(grants_data_path)

    assert len(grants_text) == 7
    assert all(text is not None for text in grants_text)

def test_predict(tmp_path):

    model_dirs = []
    individual_pred = []
    for i, vectorizer in enumerate(["count", "count", "tfidf"]):
        grant_tagger = GrantTagger(
                vectorizer_type=vectorizer,
                classifier_type="naive_bayes",
                )
        text_data.fillna('', inplace=True)
        X_train = text_data['text_1'].tolist()
        y_train = text_data['label'].tolist()
        grant_tagger.fit(X_train[1:5], y_train[1:5]) # This means count and tdidf results are different
        X_vect = grant_tagger.transform(pd.DataFrame({'Grant texts': X_train}))
        # Save results and model
        individual_pred.append(list(grant_tagger.predict(X_vect)))
        model_dir = os.path.join(tmp_path, str(i))
        grant_tagger.save_model(model_dir)
        model_dirs.append(model_dir)

    grants_data_path = os.path.join(tmp_path, 'text_data.csv')
    text_data.to_csv(grants_data_path)

    # Test that if 2 need to agree then the output will be the same
    # as one of the count vectorizer individual outputs
    ensemble_gt = EnsembleGrantTagger(
        model_dirs = model_dirs,
        grant_text_cols = ['text_1', 'text_2'],
        grant_id_col = 'id',
        num_agree = 2
        )
    grants_text = ensemble_gt.load_grants_text(grants_data_path)
    final_predictions = ensemble_gt.predict(grants_text)
    assert final_predictions == individual_pred[0]
    
    # Test that with impossibly high threshold everything is returned as 0
    ensemble_gt = EnsembleGrantTagger(
        model_dirs = model_dirs,
        grant_text_cols = ['text_1', 'text_2'],
        grant_id_col = 'id',
        num_agree = 2,
        threshold = 1.1
        )
    grants_text = ensemble_gt.load_grants_text(grants_data_path)
    final_predictions = ensemble_gt.predict(grants_text)
    assert len(final_predictions) == 7
    assert all(pred==0 for pred in final_predictions)

        