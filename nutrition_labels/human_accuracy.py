import pandas as pd

from nutrition_labels.useful_functions import pretty_confusion_matrix
from nutrition_labels.grant_data_processing import merge_tags

def clean_codes(data, col_name):
    return [str(int(a)) if not pd.isnull(a) else a for a in data[col_name]]

def evaluate_epmc(data, first_label, second_label, name_1='Nonie', name_2='Liz'):

    print(f"{name_1} labelled:")
    print(data.groupby([first_label])[first_label].count())

    print(f"{name_2} labelled:")
    print(data.groupby([second_label])[second_label].count())

    both_labelled = data.dropna(subset=[first_label, second_label])

    both_labelled.replace('4', '5', inplace=True)
    both_labelled.replace('6', '5', inplace=True)
    
    print("Proportion of times we exactly agree on tool, dataset, model, not relevant:")
    hard_agree = both_labelled[both_labelled[first_label]==both_labelled[second_label]]
    print(len(hard_agree)/len(both_labelled))
    values = list(set([i for i in both_labelled[[first_label, second_label]].values.ravel() if pd.notnull(i)]))
    print(pretty_confusion_matrix(
        both_labelled[first_label],
        both_labelled[second_label], labels=values, text=[name_1, name_2])
    )

    print("Proportion of times we agree on relevant/not relevent:")
    soft_agree = both_labelled[(
        (both_labelled[first_label]=='5') & (both_labelled[second_label]=='5')
        ) | (
        (both_labelled[first_label]!='5') & (both_labelled[second_label]!='5')
        )]
    print(len(soft_agree)/len(both_labelled))
    values = list(set([i for i in soft_agree[[first_label, second_label]].values.ravel() if pd.notnull(i)]))
    print(pretty_confusion_matrix(
        soft_agree[first_label],
        soft_agree[second_label], labels=values, text=[name_1, name_2])
    )


def evaluation_grants(data, first_label, second_label, name_1="Nonie's original", name_2="Nonie's second"):

    # original labelling
    print(f'{name_1} labelling of grant data set')
    print(len(data.dropna(subset=[first_label])))
    print(data.groupby(first_label)[first_label].count())

    # Second relabelling
    print(f'{name_2} labelling')
    print(len(data.dropna(subset=[second_label])))
    print(data.groupby(second_label)[second_label].count())

    # get only relabelled data
    data = data.dropna(subset = [first_label,second_label])

    print('Number of relabeled grants')
    print(len(data))

    full_agree = data[data[first_label] == data[second_label]]
    print('Proportion of times there was agreement')
    print(len(full_agree)/len(data))

    # Confusion matrix
    values = list(set([i for i in data[[first_label, second_label]].values.ravel() if pd.notnull(i)]))
    print(pretty_confusion_matrix(
        data[first_label],
        data[second_label],
        labels=values, text=[name_1, name_2])
    )

def incorporate_beckys_epmc_tags(liz_add_epmc, epmc_tags_query_two, becky_epmc):

    epmc_tags = pd.concat([liz_add_epmc, epmc_tags_query_two], ignore_index=True)
    epmc_tags = merge_tags(epmc_tags, 'code', 'Liz code')
    epmc_tags['pmid'] = [int(i) for i in epmc_tags['pmid']]
    becky_epmc['pmid'] = [int(i) for i in becky_epmc['pmid']]
    epmc_tags = epmc_tags.join(becky_epmc[['pmid', 'Becky code']].set_index('pmid'), on='pmid') 

    return epmc_tags


def incorporate_beckys_grants_tags(liz_add_grants, becky_grants):

    grant_tags = merge_tags(liz_add_grants, 'tool relevent ', 'Liz code')
    grant_tags = grant_tags.join(becky_grants[['Internal ID', 'Becky code']].set_index('Internal ID'), on='Internal ID') 

    return grant_tags


if __name__ == '__main__':

    liz_add_epmc = pd.read_csv(
        'data/raw/EPMC_relevant_tool_pubs_manual_edit_Lizadditions.csv',
        encoding = "latin"
        )
    epmc_tags_query_two = pd.read_csv('data/raw/EPMC_relevant_pubs_query2_manual_edit.csv')
    becky_epmc = pd.read_csv('data/processed/epmc_sample_becky.csv')

    nonie_relabel_grants = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019_manual_edit_relabeling.csv')
    liz_add_grants = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019_manual_edit_Lizadditions.csv')
    becky_grants = pd.read_csv('data/processed/grants_sample_becky.csv')

    print("\n------Evaluation for Liz and Nonie's EPMC tags------")
    liz_add_epmc['Nonie code'] = clean_codes(liz_add_epmc, 'code')
    liz_add_epmc['Liz code'] = clean_codes(liz_add_epmc, 'Liz code')
    evaluate_epmc(liz_add_epmc, 'Nonie code', 'Liz code', name_1='Nonie', name_2='Liz')

    print("\n------Evaluation for Liz/Nonie and Becky's EPMC tags------")
    epmc_tags = incorporate_beckys_epmc_tags(liz_add_epmc, epmc_tags_query_two, becky_epmc)
    epmc_tags['Becky code'] = clean_codes(epmc_tags, 'Becky code')
    epmc_tags['Merged code'] = clean_codes(epmc_tags, 'Merged code')
    evaluate_epmc(epmc_tags, 'Merged code', 'Becky code', name_1='Nonie/Liz', name_2='Becky')

    print("\n------Evaluation for Nonie's second go at grants tags------")
    evaluation_grants(nonie_relabel_grants, 'tool relevent ', 'double_check ', name_1="Nonie's original", name_2="Nonie's second")

    print("\n------Evaluation for Liz and Nonie's grants tags------")
    evaluation_grants(liz_add_grants, 'tool relevent ', 'Liz code', name_1="Nonie", name_2="Liz")

    print("\n------Evaluation for Liz/Nonie and Becky's grants tags------")
    grant_tags = incorporate_beckys_grants_tags(liz_add_grants, becky_grants)
    evaluation_grants(grant_tags, 'Merged code', 'Becky code', name_1='Nonie/Liz', name_2='Becky')


    