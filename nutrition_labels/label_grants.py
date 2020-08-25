import pandas as pd
import numpy as np
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
    # relevant_grants = []
    # for i, row in grant_data.iterrows():
    #    if row['Relevance code'] == 1.0:
    #        relevant_grants.append({'Internal ID': row['Internal ID'], 'Found by': 'Manually tagged'})
    #    elif row['Predicted relevance code'] == 1.0:
    #        relevant_grants.append({'Internal ID': row['Internal ID'], 'Found by': 'Model prediction'})

    # 642 grants
    # relevant_grants = pd.DataFrame(relevant_grants)

    grant_data['Final label'] = (grant_data[['Relevance code','Predicted relevance code']]
                                 .apply(lambda x: x[1] if np.isnan(x[0]) else x[0], axis = 1))

    grant_data['Found by'] = (grant_data[['Relevance code', 'Predicted relevance code']]
                             .apply(lambda x: 'Model prediction' if np.isnan(x[0]) else 'Manually tagged', axis = 1))

    grant_data_out = grant_data[['Internal ID','Relevance code', 'Internal ID 6 digit','Predicted relevance code',
                                 'Final label', 'Found by']]
    relevant_grants = grant_data_out[grant_data_out['Final label'] == 1]

    if write_out == True:
        relevant_grants.to_csv(f'data/processed/{model_name}_relevant_grant_list.csv')
        grant_data_out.to_csv(f'data/processed/{model_name}_all_grant_list.csv')

    print(relevant_grants['Internal ID'].tolist())
    return grant_data_out


def get_test_results(file,f1_cutoff = 0,precision_cutoff = 0,recall_cutoff = 0):
    f = open('models/' + file + "/training_information.txt", "r")
    lines = f.readlines()
    f.close()
    results = lines[7]
    results = re.sub('Test scores: ', '', results)
    results_dict = ast.literal_eval(results)
    if (results_dict['f1'] >= f1_cutoff and results_dict['precision_score'] >= precision_cutoff and results_dict['recall_score'] >= recall_cutoff):
        return True
    else:
        return False

if __name__ == '__main__':

    training_data = pd.read_csv('data/processed/training_data.csv')
    grant_data = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019.csv')

    model_dirs = os.listdir('models')

    useful_models = [i for i in model_dirs if get_test_results(i,0.8,0.82,0.82)]

    model_results = [label_grants(i,grant_data,training_data) for i in useful_models]

    cutoff = len(useful_models)

    for indx,i in enumerate(model_results):
        if indx == 0:
            results_df = i[['Internal ID','Final label', 'Found by']]
            results_df = results_df.rename()




