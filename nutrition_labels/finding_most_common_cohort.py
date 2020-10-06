import pandas as pd
import random
from nutrition_labels.client import EPMCClient


cohorts = pd.read_csv('data/processed/health_data_sets_list_manual_edit.csv')

def get_cites(pub_id):
    client = EPMCClient()
    session = client.requests_session()
    citations = client.get_citations(session,pub_id)
    year_list = [i['pubYear'] for i in citations]
    recent_list = [i for i in year_list if i > 2014]
    total = len(year_list)
    recent = len(recent_list)
    return [total,recent]


cite_values = [get_cites(i) for i in cohorts['id']]
cohorts['total cites'] = [i[0] for i in cite_values]
cohorts['recent cites'] = [i[1] for i in cite_values]
cohorts = cohorts.sort_values(by = 'recent cites', ascending= False)