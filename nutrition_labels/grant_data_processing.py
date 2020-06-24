import pandas as pd
import re
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

grant_ref = grant_data[['Internal ID','Description','Award Date']]
grant_ref['grant_number'] = grant_data['Internal ID'].apply(lambda x: re.sub('/.*','',x))
grant_ref = pd.merge(grant_ref,epmc_df, how = 'inner', on = 'grant_number')
grant_ref['Description'] = grant_ref['Description'].apply(lambda x: re.sub('\<.*\>{1:}','',x))
grant_ref = grant_ref.drop_duplicates(subset=['Description','grant_number','pmid'])
grant_duplicates = grant_ref[grant_ref.duplicated(subset=['pmid','grant_number'], keep = False)]


no_ref = epmc_df[~epmc_df['grant_number'].isin(grant_ref['grant_number'].to_list())]
# cleaning