import pandas as pd

from nutrition_labels.utils import pretty_confusion_matrix
from nutrition_labels.grant_data_processing import merge_tags

def clean_codes(data, col_name):
    return [str(int(a)) if not pd.isnull(a) else a for a in data[col_name]]

def evaluate_epmc(data, first_label, second_label, name_1='Nonie', name_2='Liz', print_all=True):

    if print_all:
        print(f"{name_1} labelled:")
        print(data.groupby([first_label])[first_label].count())

        print(f"{name_2} labelled:")
        print(data.groupby([second_label])[second_label].count())

    both_labelled = data.copy().dropna(subset=[first_label, second_label])

    both_labelled.replace('4', '5', inplace=True)
    both_labelled.replace('6', '5', inplace=True)
    
    print("Proportion of times we exactly agree on tool, dataset, model, not relevant:")
    hard_agree = both_labelled.loc[both_labelled[first_label]==both_labelled[second_label]]
    print(f"{len(hard_agree)/len(both_labelled)} out of {len(both_labelled)} we both labelled")
    
    if print_all:
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
    print(f"{len(soft_agree)/len(both_labelled)} out of {len(both_labelled)} we both labelled")
    
    if print_all:
        values = list(set([i for i in soft_agree[[first_label, second_label]].values.ravel() if pd.notnull(i)]))
        print(pretty_confusion_matrix(
            soft_agree[first_label],
            soft_agree[second_label], labels=values, text=[name_1, name_2])
        )


def evaluation_grants(data, first_label, second_label, name_1="Nonie's original", name_2="Nonie's second", print_all=True):

    if print_all:
        # original labelling
        print(f'{name_1} labelling of grant data set')
        print(len(data.dropna(subset=[first_label])))
        print(data.groupby(first_label)[first_label].count())

        # Second relabelling
        print(f'{name_2} labelling')
        print(len(data.dropna(subset=[second_label])))
        print(data.groupby(second_label)[second_label].count())

    data.replace('4', '5', inplace=True)
    # get only relabelled data
    data = data.dropna(subset = [first_label,second_label])

    full_agree = data[data[first_label] == data[second_label]]
    print('Proportion of times there was agreement')
    print(f'{len(full_agree)/len(data)} out of {len(data)} relabeled grants')

    # Confusion matrix
    values = list(set([i for i in data[[first_label, second_label]].values.ravel() if pd.notnull(i)]))
    if print_all:
        print(pretty_confusion_matrix(
            data[first_label],
            data[second_label],
            labels=values, text=[name_1, name_2])
        )


def incorporate_beckys_grants_tags(liz_add_grants, becky_grants):

    grant_tags = merge_tags(liz_add_grants, ['tool relevent ', 'Liz code'])
    grant_tags.replace(4, 5, inplace=True)
    grant_tags = grant_tags.join(becky_grants[['Internal ID', 'Becky code']].set_index('Internal ID'), on='Internal ID') 

    return grant_tags


if __name__ == '__main__':

    epmc_tags_query_one = pd.read_csv('data/raw/EPMC_relevant_tool_pubs_3082020.csv', encoding = "latin")
    epmc_tags_query_two = pd.read_csv('data/raw/EPMC_relevant_pubs_query2_3082020.csv')
    epmc_tags = pd.concat([epmc_tags_query_one, epmc_tags_query_two], ignore_index=True)

    nonie_relabel_grants = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019_manual_edit_relabeling.csv')
    liz_add_grants = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019_manual_edit_Lizadditions.csv')
    becky_grants = pd.read_csv('data/processed/grants_sample_becky.csv')

    print("\n------Evaluation for Liz and Nonie's EPMC tags------")
    epmc_tags['Nonie code'] = clean_codes(epmc_tags, 'Nonie code')
    epmc_tags['Liz code'] = clean_codes(epmc_tags, 'Liz code')
    evaluate_epmc(epmc_tags, 'Nonie code', 'Liz code', name_1='Nonie', name_2='Liz', print_all=True)

    print("\n------Evaluation for Liz/Nonie and Becky's EPMC tags------")
    epmc_tags = merge_tags(epmc_tags, ['Nonie code', 'Liz code'])
    epmc_tags['Becky code'] = clean_codes(epmc_tags, 'Becky code')
    epmc_tags['Merged code'] = clean_codes(epmc_tags, 'Merged code')
    evaluate_epmc(epmc_tags, 'Merged code', 'Becky code', name_1='Nonie/Liz', name_2='Becky', print_all=True)

    print("\n------Evaluation for Nonie's second go at grants tags------")
    evaluation_grants(nonie_relabel_grants, 'tool relevent ', 'double_check ', name_1="Nonie's original", name_2="Nonie's second")

    print("\n------Evaluation for Liz and Nonie's grants tags------")
    evaluation_grants(liz_add_grants, 'tool relevent ', 'Liz code', name_1="Nonie", name_2="Liz")

    print("\n------Evaluation for Liz/Nonie and Becky's grants tags------")
    grant_tags = incorporate_beckys_grants_tags(liz_add_grants, becky_grants)
    evaluation_grants(grant_tags, 'Merged code', 'Becky code', name_1='Nonie/Liz', name_2='Becky')

    