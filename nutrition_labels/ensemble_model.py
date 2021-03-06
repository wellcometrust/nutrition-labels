"""
1. Ensemble model to predict unseen grants data
2. Calculate metrics for ensemble model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report,  f1_score, precision_score, recall_score

import os
import re
import ast
from datetime import datetime

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.utils import pretty_confusion_matrix, clean_grants_data

class EnsembleModel():
    def __init__(
        self,
        f1_cutoff,
        precision_cutoff,
        recall_cutoff,
        before_date,
        after_date):
        self.f1_cutoff = f1_cutoff
        self.precision_cutoff = precision_cutoff
        self.recall_cutoff = recall_cutoff
        self.before_date = before_date # Inclusive
        self.after_date = after_date # Inclusive

    def find_useful_models(self):
        """
        Find models from the models folder which have dates between (inclusive) before_date & after_date
        and have metrics >= the cutoffs
        """

        model_dirs = os.listdir('models')
        model_dirs.remove('.DS_Store')
        model_dirs = [model_dir for model_dir in model_dirs if (
                model_dir[-6:].isnumeric()) and (
                    int(model_dir[-6:]) in range(self.after_date, self.before_date+1)
                    )
                ]

        useful_models = [i for i in model_dirs if get_test_results(i, self.f1_cutoff, self.precision_cutoff, self.recall_cutoff)]

        print(f'{len(useful_models)} useful models found')

        return useful_models

    def predict(self, texts, useful_models, num_agree=1):
        """
        Predict whether grant texts (list) are tech grants or not 
        using an agreement of num_agree (int)
        of the models in useful_models (list)
        """
        if num_agree > len(useful_models):
            num_agree = len(useful_models)

        model_predictions = {}
        for model_name in useful_models:
            print(f'Predicting for {model_name}...')
            model_predictions[f'{model_name} predictions'] = label_grants(model_name, texts)

        model_predictions_df = pd.DataFrame(model_predictions)
        prediction_sums = model_predictions_df.sum(axis=1)
        model_predictions_df['Number of models agree tech grant'] = prediction_sums
        model_predictions_df['Ensemble prediction'] = (prediction_sums >= num_agree).astype(int)
        self.model_predictions_df = model_predictions_df

        return model_predictions_df['Ensemble prediction'].tolist()
        
def process_grants_data(grant_data, training_data, split_seed):

    # Label which training data was in the original test/train sets
    # Just need to separate the internal IDs by the same indexes
    # We use vectorizer type bert since this option deals with the X data indices appropriately
    # for using training_data['Internal ID'] in place of the usual X_vect,
    # otherwise the indices selection for the sampling done in 'split_data' are incorrect for X
    grant_tagger = GrantTagger(relevant_sample_ratio=1, vectorizer_type='bert')
    train_data, test_data, _ = grant_tagger.split_data(
        training_data['Internal ID'],
        training_data['Relevance code'],
        split_seed=split_seed)

    (x_train, _, _) = train_data
    (x_test, _, _) = test_data

    # Add the manually tagged relevance code label from the training data
    grant_data = pd.merge(
        grant_data,
        training_data[['Internal ID', 'Relevance code']],
        how = 'left',
        on = ['Internal ID']
        )

    # Process grants data for predicting
    grant_data = clean_grants_data(grant_data)
    grant_data['Grant texts'] = grant_data[['Title', 'Grant Programme:Title', 'Description']].agg(
                '. '.join, axis=1
                ).tolist()

    # You don't need all the columns
    grant_data = grant_data[['Internal ID', 'Relevance code', 'Grant texts']]

    data_trained =[]
    for ref in grant_data['Internal ID']:
        if ref in x_train:
            data_trained.append('Training data')
        elif ref in x_test:
            data_trained.append('Test data')
        else:
            data_trained.append('Unseen data')
    grant_data['How has this grant been used before?'] = data_trained

    return grant_data

def label_grants(model_name, grants_text, write_out = False):

    # Loading a trained model and vectorizer to predict on all the grants data:
    grant_tagger_loaded = GrantTagger()
    grant_tagger_loaded.load_model(os.path.join('models', model_name))

    new_grants_vect = grant_tagger_loaded.vectorizer.transform(grants_text)
    predictions = grant_tagger_loaded.predict(new_grants_vect)

    return predictions


def get_test_results(file,f1_cutoff = 0,precision_cutoff = 0,recall_cutoff = 0):
    with open('models/' + file + "/training_information.txt", "r") as f:
        for line in f.readlines():
            if 'Test scores: ' in line:
                results = line
    results = re.sub('Test scores: ', '', results)
    results_dict = ast.literal_eval(results)
    if (results_dict['f1'] >= f1_cutoff and results_dict['precision_score'] >= precision_cutoff and results_dict['recall_score'] >= recall_cutoff):
        return True
    else:
        return False

def get_seed_results(file):
    with open('models/' + file + "/training_information.txt", "r") as f:
        for line in f.readlines():
            if 'Split random seed' in line:
                seed = line
    seed = re.sub('Split random seed: ', '', seed)
    return ast.literal_eval(seed)

def evaluate_ensemble(grant_data, pred_col='Ensemble predictions'):
    test_grants = grant_data.loc[grant_data['How has this grant been used before?']=='Test data']

    y = test_grants['Relevance code'].tolist()
    y_predict = test_grants[pred_col].tolist()
    # Evaluate ensemble results
    scores = {
            'accuracy': accuracy_score(y, y_predict),
            'f1': f1_score(y, y_predict, average='binary'),
            'precision_score': precision_score(y, y_predict, zero_division=0, average='binary'),
            'recall_score': recall_score(y, y_predict, zero_division=0, average='binary'),
            'Test classification report': classification_report(y, y_predict),
            'Test confusion matrix': pretty_confusion_matrix(y, y_predict)}
    return scores

if __name__ == '__main__':

    training_file = 'data/processed/training_data/210126/training_data.csv'

    training_data = pd.read_csv(training_file)
    grant_data = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019.csv')

    datestamp = datetime.now().date().strftime('%y%m%d')

    f1_cutoff = 0.8
    precision_cutoff = 0.8
    recall_cutoff = 0.8
    after_date = 210128
    before_date = 210128

    ensemble_model = EnsembleModel(
        f1_cutoff =f1_cutoff,
        precision_cutoff = precision_cutoff,
        recall_cutoff = recall_cutoff,
        before_date = before_date,
        after_date = after_date)

    useful_models = ensemble_model.find_useful_models()

    split_seed = [get_seed_results(model_dir) for model_dir in useful_models]
    print(f'There is/are {len(set(split_seed))} unique split seeds used for these models '\
        'if this is more than 1 then the ensemble model metrics can be ignored')
    split_seed = split_seed[0]

    grant_data = process_grants_data(grant_data, training_data, split_seed)

    # Predict for each model
    grants_text = grant_data['Grant texts'].tolist()
  
    _ = ensemble_model.predict(grants_text, useful_models)
    model_predictions_df = ensemble_model.model_predictions_df
    del model_predictions_df['Ensemble prediction']
    grant_data = pd.concat([grant_data, model_predictions_df], axis=1)

    # drop the 'Grant texts' column as no need to save this
    del grant_data['Grant texts']

    prediction_sums = model_predictions_df['Number of models agree tech grant']

    # Calculate the different final predictions and scores for different cutoffs

    for cutoff in list(range(1, len(useful_models)+1)):
        grant_data[f'Ensemble predictions - {cutoff} models'] = [1 if pred_sum >= cutoff else 0 for pred_sum in prediction_sums]
        grant_data[f'Final ensemble label - {cutoff} models'] = (grant_data[['Relevance code', f'Ensemble predictions - {cutoff} models']]
                                     .apply(lambda x: x[1] if np.isnan(x[0]) else x[0], axis = 1))
        grant_data[f'Final ensemble label found by - {cutoff} models'] = (grant_data[['Relevance code', f'Ensemble predictions - {cutoff} models']]
                                 .apply(lambda x: 'Model prediction' if np.isnan(x[0]) else 'Manually tagged', axis = 1))
        relevant_grants = grant_data[grant_data[f'Final ensemble label - {cutoff} models'] == 1]
        print(f"Found {len(relevant_grants)} relevant grants")
        scores = evaluate_ensemble(grant_data, pred_col=f'Ensemble predictions - {cutoff} models')
        # Save
        output_path = f'data/processed/ensemble/{datestamp}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, f'{datestamp}_training_information_{cutoff}models.txt'), 'w') as f:
            f.write('Ensemble models: ' + str(useful_models))
            f.write('\nf1_cutoff: ' + str(f1_cutoff))
            f.write('\nprecision_cutoff: ' + str(precision_cutoff))
            f.write('\nrecall_cutoff: ' + str(recall_cutoff))
            f.write('\nNumber relevant grants: ' + str(len(relevant_grants)))
            f.write('\nOnly includes models after: '+ str(after_date))
            f.write('\ntraining_file: '+ training_file)
            for key, value in scores.items():
                f.write('\n' + key + ': ' + str(value))
        relevant_grants.to_csv(os.path.join(output_path, f'{datestamp}_ensemble_results_{cutoff}models.csv'), index = False)

    grant_data.to_csv(os.path.join(output_path, f'{datestamp}_all_ensemble_results.csv'), index = False)
