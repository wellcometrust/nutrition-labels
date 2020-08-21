import pandas as pd

import os
import re
import ast

from nutrition_labels.grant_tagger import GrantTagger
from nutrition_labels.grant_data_processing import clean_grants_data


def label_grants (model_name, grant_data, training_data, write_out = False):
# Add the manually tagged relevance code label from the training data
    grant_data = pd.merge(
        grant_data,
        training_data[['Internal ID', 'Relevance code']],
        how = 'left',
        on = ['Internal ID']
        )

    # Process grants data for predicting
    grant_data = clean_grants_data(grant_data)
    grants_text = grant_data[['Title', 'Grant Programme:Title', 'Description']].agg(
                '. '.join, axis=1
                ).tolist()

    # Loading a trained model and vectorizer to predict on all the grants data:
    grant_tagger_loaded = GrantTagger()
    grant_tagger_loaded.load_model(os.path.join('models', model_name))

    new_grants_vect = grant_tagger_loaded.vectorizer.transform(grants_text)
    predictions = grant_tagger_loaded.predict(new_grants_vect)
    grant_data['Predicted relevance code'] = predictions

    # Produce a list of relevant grants -
    # if manually tagged use this, if not, use the model's prediction
    relevant_grants = []
    for i, row in grant_data.iterrows():
        if row['Relevance code'] == 1.0:
            relevant_grants.append({'Internal ID': row['Internal ID'], 'Found by': 'Manually tagged'})
        elif row['Predicted relevance code'] == 1.0:
            relevant_grants.append({'Internal ID': row['Internal ID'], 'Found by': 'Model prediction'})

    # 642 grants
    relevant_grants = pd.DataFrame(relevant_grants)

    if write_out == True:
        relevant_grants.to_csv(f'data/processed/{model_name}_grant_list.csv')

    print(relevant_grants['Internal ID'].tolist())
    return relevant_grants['Internal ID'].tolist()


def get_test_f1(file):
    f = open('models/' + file + "/training_information.txt", "r")
    lines = f.readlines()
    f.close()
    results = lines[7]
    results = re.sub('Test scores: ', '', results)
    results_dict = ast.literal_eval(results)
    return (results_dict['f1'])

if __name__ == '__main__':

    training_data = pd.read_csv('data/processed/training_data.csv')
    grant_data = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019.csv')

    model_dirs = os.listdir('models')

    useful_models = [i for i in model_dirs if get_test_f1(i) >= 0.85]

    model_results = {i:label_grants(i,grant_data,training_data) for i in useful_models}

    for i in useful_models:
        print(i + ': '+ str(len(model_results[i])))

    results_list = [i for v in model_results.values() for i in v]
    results_set = set(results_list)

    cutoff = len(useful_models)

    results_count = [i for i in results_set if results_list.count(i) >= cutoff]


