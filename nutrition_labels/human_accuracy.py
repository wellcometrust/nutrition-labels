import pandas as pd
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    liz_add_epmc = pd.read_csv(
        'data/raw/EPMC_relevant_tool_pubs_manual_edit_Lizadditions.csv',
        encoding = "latin"
        )

    str_codes = [str(int(a)) if not pd.isnull(a) else a for a in liz_add_epmc['code']]
    liz_add_epmc['Nonie code'] = [a if a!='55' else '5' for a in str_codes]

    print("Nonie labelled:")
    print(liz_add_epmc.groupby(['Nonie code'])['Nonie code'].count())

    print("Liz labelled:")
    print(liz_add_epmc.groupby(['Liz code'])['Liz code'].count())

    both_labelled = liz_add_epmc.dropna(subset=['Nonie code', 'Liz code'])

    print("Proportion of times we exactly agree on tool, dataset, model, not relevant")
    hard_agree = both_labelled[both_labelled['Nonie code']==both_labelled['Liz code']]
    print(len(hard_agree)/len(both_labelled))
    # print(hard_agree[['Nonie code', 'Liz code']])

    print("Proportion of times we agree on relevant/not relevent")
    soft_agree = both_labelled[(
        (both_labelled['Nonie code']=='5') & (both_labelled['Liz code']=='5')
        ) | (
        (both_labelled['Nonie code']!='5') & (both_labelled['Liz code']!='5')
        )]
    print(len(soft_agree)/len(both_labelled))
    print(soft_agree[['Nonie code', 'Liz code']])

    # find what got labelled as what
    conf_matrix = pd.DataFrame(confusion_matrix(both_labelled['Nonie code'], both_labelled['Liz code']),
                               columns=list(range(1, 7)),
                               index=list(range(1, 7)))
    print('confusion matrix of Liz labels and Nonie lables')
    print(conf_matrix)

    # Nonie relabelling of grant data
    grant_relabeling = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019_manual_edit_relabeling.csv')

    # get only labelled data
    grant_relabeling = grant_relabeling.dropna(subset = ['tool relevent ','double_check '])

