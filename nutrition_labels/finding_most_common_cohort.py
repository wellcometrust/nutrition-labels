import pandas as pd
from datalabs.epmc.client import EPMCClient


cohorts = pd.read_csv('data/processed/health_data_sets_list_manual_edit.csv')

def get_num_citations(pub_id):
    client = EPMCClient()
    session = client.requests_session()
    citations = client.get_citations(session,pub_id)
    recent_list =  [i['pubYear'] for i in citations if i['pubYear'] > 2014]
    total = len(citations)
    recent = len(recent_list)
    return [total,recent]

if __name__ == '__main__':

    cite_values = [get_num_citations(i) for i in cohorts['id']]
    cohorts['total number of citations'] = [i[0] for i in cite_values]
    cohorts['recent number of citations'] = [i[1] for i in cite_values]
    cohorts = cohorts.sort_values(by = 'recent number of citations', ascending= False)
    cohorts.to_csv('data/processed/health_data_reference_count.csv',index=False)