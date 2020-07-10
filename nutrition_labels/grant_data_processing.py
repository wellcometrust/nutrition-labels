import pandas as pd
import re
from nutrition_labels.useful_functions import remove_useless_string, only_text

# load data
epmc_tags = pd.read_csv('data/raw/EPMC_relevant_tool_pubs_manual_edit.csv')
rf_tags = pd.read_csv('data/raw/ResearchFish/research_fish_manual_edit.csv')
grant_tags = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019_manual_edit.csv')
grant_data = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019.csv')

# clean data
rf_tags = rf_tags.dropna(subset = ['code '])
epmc_tags = epmc_tags.dropna(subset = ['code','WTgrants'])
grant_tags = grant_tags.dropna(subset = ['tool relevent '])

epmc_cols = ['pmid','code']
epmc_list = []
for i,row in epmc_tags.iterrows():
    grant_num = row['WTgrants']
    if len(grant_num) == 5:
        grant_num = ['0' + grant_num]
    elif len(grant_num) > 6:
        count = 1
        merged_grant_chunk = ''
        merge_grant_nums = list()
        for grant_chunk in grant_num.split(','):
            if len(grant_chunk) ==2:
                grant_chunk = '0' + grant_chunk
            if count == 2:
                merge_grant_nums.append(merged_grant_chunk + grant_chunk)
                merged_grant_chunk = ''
                count = 1
            elif count == 1:
                merged_grant_chunk = grant_chunk
                count = count + 1
        merge_grant_nums = [nums for nums in merge_grant_nums if nums != '000000']
        grant_num = merge_grant_nums
    else:
        grant_num = [grant_num]
    for num in grant_num:
        epmc_list.append({'pmid':row['pmid'],
                          'code':row['code'],
                          'grant_number':num})

epmc_df = pd.DataFrame(epmc_list)
epmc_df = epmc_df[epmc_df['code'].isin([1,2,3])] # Only selecting useful codes

# getting WT grant number


grant_ref = grant_data[['Grant Programme:Title','Internal ID','Description','Award Date']]
grant_ref['grant_number'] = grant_data['Internal ID'].apply(lambda x: re.sub('/.*','',x))
grant_ref = pd.merge(grant_ref,epmc_df, how = 'inner', on = 'grant_number')
grant_ref['Description'] = grant_ref['Description'].apply(remove_useless_string)
grant_ref['Description'] = grant_ref['Description'].apply(only_text)
grant_ref = grant_ref.drop_duplicates(subset=['Description','grant_number','pmid'])
grant_ref = grant_ref.sort_values('Award Date', ascending=False).drop_duplicates(subset = ['grant_number', 'pmid'])

# finding abstract without a matching code
no_ref = epmc_df[~epmc_df['grant_number'].isin(grant_ref['grant_number'].to_list())]
no_ref = no_ref[~no_ref['pmid'].isin(grant_ref['pmid'].to_list())]
missing_abstracts = epmc_tags[epmc_tags['pmid'].isin(no_ref['pmid'].to_list())] #I cant find where this grant is at all

# get list
code_list = grant_ref[['Internal ID','code','pmid']]

# get rf data
rf_useful = rf_tags[rf_tags['code '].isin([1,2,3])]
rf_useful = rf_useful[['code ','Grant Reference','Name']]
rf_useful = rf_useful.rename({'code ':'code','Grant Reference':'Internal ID','Name':'pmid'}, axis = 1)

print("Grants flagged as relevant in EPMC publications not found in grants data: ")
print(set(epmc_df['grant_number']).difference(set(grant_ref['grant_number'])))
print("PMIDs flagged as relevant in Wellcome EPMC publications but not found to link to any grants in the grants data: ")
print(set(epmc_df['pmid']).difference(set(grant_ref['pmid'])))

# merge abstract and rfs
code_list = pd.concat([code_list,rf_useful])
code_list['Internal ID'] = code_list['Internal ID'].str.strip(' ')
code_list_final = pd.merge(grant_data[['Internal ID','Title','Description']],code_list, how = 'inner', on = 'Internal ID')
rf_not_in = code_list[~code_list['Internal ID'].isin(code_list_final['Internal ID'])] # six codes arent in the wellcome data

print("Grants flagged as relevant in RF data not found in grants data: ")
print(set(code_list['Internal ID']).difference(set(code_list_final['Internal ID'])))
# getting control data from wellcome grant data
# 3== not that relevent 4==not at all relevent
needed_cols = ['Internal ID','Title','Description','tool relevent ']
grant_code = grant_tags.loc[grant_tags['tool relevent ']==5.0,needed_cols]
grant_code = grant_code.rename({'tool relevent ':'code'},axis = 1)

# final list
code_list_final = pd.concat([code_list_final,grant_code])
code_list_final = code_list_final.drop_duplicates('Internal ID')
code_list_final.to_csv('data/processed/training_data.csv', index = False)