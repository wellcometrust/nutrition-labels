import re

import pandas as pd
import numpy as np

from useful_functions import remove_useless_string, only_text

def merge_tags(data, nonie_col_name, liz_col_name):
    # If Nonie and Liz have labelled it use Nonies
    data = data.dropna(subset=[nonie_col_name, liz_col_name], how='all').reset_index(drop=True)
    data['Merged code'] = [row[liz_col_name] if pd.isnull(row[nonie_col_name]) else row[nonie_col_name] for _, row in data.iterrows()]
    
    return data

def process_epmc(epmc_tags_query_one, epmc_tags_query_two, epmc_code_dict):

    # Merge EPMC data and normalise the codes
    # If Nonie and Liz have labelled it use Nonies
    epmc_tags = pd.concat([epmc_tags_query_one, epmc_tags_query_two], ignore_index=True)
    epmc_tags = merge_tags(epmc_tags, 'code', 'Liz code')
    epmc_tags['Normalised code'] = [epmc_code_dict[str(int(code))] for code in epmc_tags['Merged code']]

    # No need to include tags if no grant number is given or you don't want to include
    # the grant in the training data
    epmc_tags.dropna(subset=['WTgrants', 'Normalised code'], inplace=True)

    # Create list of dicts for each WT grant number given
    epmc_list = []
    for i, row in epmc_tags.iterrows():
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
                              'Normalised code - EPMC':int(row['Normalised code']),
                              'Internal ID 6 digit':num,})

    epmc_df = pd.DataFrame(epmc_list)
    epmc_df.drop_duplicates(subset=['Internal ID 6 digit', 'Normalised code - EPMC'], inplace=True)
    
    return epmc_df

def process_RF(rf_tags, rf_code_dict):

    rf_tags.dropna(subset=['code '], inplace=True)
    rf_tags['Normalised code'] = [rf_code_dict[str(int(code))] for code in rf_tags['code ']]

    # No need to include tags if no grant number is given or you don't want to include
    # the grant in the training data
    rf_tags.dropna(subset=['Grant Reference', 'Normalised code'], inplace=True)

    # Create list of dicts for each WT grant number given
    rf_list = []
    for i, row in rf_tags.iterrows():
        # Strip whitespace from grant ref, since sometimes it has it
        rf_list.append({
            'RF question':row['file'],
            'RF Name':row['Name'],
            'Normalised code - RF':int(row['Normalised code']),
            'Internal ID':row['Grant Reference'].strip(),
            })

    rf_df = pd.DataFrame(rf_list)
    rf_df.drop_duplicates(subset=['Internal ID', 'Normalised code - RF'], inplace=True)

    return rf_df

def process_grants(grant_tags, grants_code_dict):

    # If Nonie and Liz have labelled it use Nonies
    grant_tags = merge_tags(grant_tags, 'tool relevent ', 'Liz code')
    grant_tags['Normalised code'] = [grants_code_dict[str(int(code))] for code in grant_tags['Merged code']]

    # No need to include tags if you don't want to include
    # the grant in the training data
    grant_tags.dropna(subset=['Normalised code'], inplace=True)

    # Create list of dicts for each WT grant number given
    grants_list = []
    for i, row in grant_tags.iterrows():
        # Strip whitespace from grant ref, since sometimes it has it
        grants_list.append({
            'Normalised code - grants':int(row['Normalised code']),
            'Internal ID':row['Internal ID'],
            })
    grants_df = pd.DataFrame(grants_list)
    grants_df.drop_duplicates(subset=['Internal ID', 'Normalised code - grants'], inplace=True)

    return grants_df

if __name__ == '__main__':
    
    # load data
    epmc_tags_query_one = pd.read_csv('data/raw/EPMC_relevant_tool_pubs_manual_edit_Lizadditions.csv', encoding = "latin")
    epmc_tags_query_two = pd.read_csv('data/raw/EPMC_relevant_pubs_query2_manual_edit.csv')
    rf_tags = pd.read_csv('data/raw/ResearchFish/research_fish_manual_edit.csv')
    grant_tags = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019_manual_edit_Lizadditions.csv')
    grant_data = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019.csv')

    # Normalising the codes so they are all similar for different data sources
    # 'None' if you dont want to include these tags in the training data
    epmc_code_dict = {'1': 1, '2': 2, '3': 3, '4': None, '5': None, '6': None}
    rf_code_dict = {'1': 1, '2': 2, '3': 3, '4': None, '5': None}
    grants_code_dict = {'1': 1, '4': None, '5': 5}

    # Process each of the 3 data sources separately and output a 
    # dataframe for each of grant numbers - cleaned tags links
    epmc_df = process_epmc(epmc_tags_query_one, epmc_tags_query_two, epmc_code_dict)
    rf_df = process_RF(rf_tags, rf_code_dict)
    grants_df = process_grants(grant_tags, grants_code_dict)

    print('Tagged data to include from EPMC:')
    print(len(epmc_df))
    print('Tagged data to include from RF:')
    print(len(rf_df))
    print('Tagged data to include from grant descriptions:')
    print(len(grants_df))

    # Clean grant descriptions of html and remove any duplicates
    grant_data['Description'] = grant_data['Description'].apply(remove_useless_string)
    grant_data.dropna(subset=['Description'], inplace=True)
    grant_data.drop_duplicates('Internal ID', inplace=True)
    grant_data['Internal ID 6 digit'] = grant_data['Internal ID'].apply(lambda x: re.sub('/.*','',x))

    # Link with RF data (which uses 13 digit)
    grant_data = pd.merge(grant_data, rf_df, how = 'left', on = 'Internal ID')
    # Link with grant data (which uses 13 digit)
    grant_data = pd.merge(grant_data, grants_df, how = 'left', on = 'Internal ID')

    # Order grants by 1. those that have tags from RF or grants,
    # 2. if they havent been tagged yet, those with the most recent award date should go higher up
    grant_data.sort_values(by=['Normalised code - RF', 'Normalised code - grants', 'Award Date'], ascending=False, inplace=True)

    # Link with EPMC data (which uses 6 digit)
    grant_data = pd.merge(grant_data, epmc_df, how = 'left', on = 'Internal ID 6 digit')

    print("Grants flagged as relevant in EPMC publications not found in grants data: ")
    print(set(epmc_df['Internal ID 6 digit']).difference(set(grant_data['Internal ID 6 digit'])))
    print("PMIDs flagged as relevant in Wellcome EPMC publications but not found to link to any grants in the grants data: ")
    print(set(epmc_df['pmid']).difference(set(grant_data['pmid'])))
    print("Grants flagged as relevant in RF data not found in grants data: ")
    print(set(rf_df['Internal ID']).difference(set(grant_data['Internal ID'])))

    # Final list
    grant_data = grant_data.dropna(
        subset=['Normalised code - RF', 'Normalised code - grants', 'Normalised code - EPMC'], 
        how='all').reset_index(drop=True)
    # Get the final single code for each grant:
    # If there is conflict in tags, the order of truth is:
    # 1. EPMC (this shows solid evidence of the output)
    # 2. RF (they might not have created the output at the time of the report)
    # 3. Grants (they might not have created the output at the time of the application)
    code = []
    for _, row in grant_data.iterrows():
        if pd.notnull(row['Normalised code - EPMC']):
            code.append(row['Normalised code - EPMC'])
        elif pd.notnull(row['Normalised code - RF']):
            code.append(row['Normalised code - RF'])
        elif pd.notnull(row['Normalised code - grants']):
            code.append(row['Normalised code - grants'])
        else:
            print('Should have deduplicated this row?')

    # Map the final codes onto whether the grant is 'relevant' (1) or not (0)
    relevance_dict = {1:1, 2:1, 3:1, 5:0}
    grant_data['Relevance code'] = [relevance_dict[int(c)] for c in code]

    grant_data = grant_data[[
        'Internal ID', 'RF question', 'RF Name', 'pmid', 'Relevance code',
        'Normalised code - RF', 'Normalised code - grants', 'Normalised code - EPMC',
        'Description', 'Title', 'Grant Programme:Title']]
    grant_data.to_csv('data/processed/training_data.csv', index = False)
