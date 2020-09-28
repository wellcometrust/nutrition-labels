import json
import pandas as pd

if __name__ == '__main__'

    # Read in data

    data = []
    for row in open('data/processed/tech_grants_references_dict_2020-09-16.json', "r"):
       data.append(json.loads(row))


    with open('data/processed/tech_grants_reference_ids_counter_2020-09-16.json') as json_file:
        count = json.load(json_file)


    references_df = pd.DataFrame(data)

    count_df = pd.DataFrame({'id':[k for k,v in count.items()],'count':[v for k,v in count.items()]},)

    # merge references with count

    count_df = pd.merge(count_df,references_df, 'left', 'id')
    count_df = count_df.drop(['source', 'citationType', 'authorString',
           'journalAbbreviation', 'issue', 'pubYear', 'volume', 'pageInfo',
           'citedOrder', 'match', 'essn', 'issn', 'publicationTitle'], axis = 1).sort_values(by = 'count', ascending= False)
    count_df = count_df[count_df['count'] > 20]

    count_df.to_csv('data/processed/reference_counter.csv',index=False)

    # get cohort profiles

    data_set_terms = ['cohort','profile','resource','biobank','Cohort','Profile','Biobank','Resource']

    count_df['data set'] = count_df['title'].apply(lambda x: 1 if any(i in x for i in data_set_terms ) else 0)
    data_sets = count_df[count_df['data set'] == 1]
    data_sets.to_csv('data/processed/health_data_sets_list.csv', index= False)


