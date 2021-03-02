"""
Small script to take EPMC data and create a dictionary of pmid: list of grants

This is because the manually labelled (in Excel) data becomes slightly corrupted:
e.g. in the original data the list of grants
086151,104104,104104
becomes
86,151,104,104,104,100
due to Excel processing it as a number not a string and rounding.
The rounding caused the wrong grants linked or grants not being found.
"""

import json

import pandas as pd

# Original EPMC data from fortytwo (not manually labelled)
epmc_query_one = pd.read_csv('data/raw/EPMC/EPMC_relevant_tool_pubs.csv')
epmc_query_two = pd.read_csv('data/raw/EPMC/EPMC_relevant_pubs_query2.csv')

epmc_data = pd.concat([epmc_query_one, epmc_query_two], ignore_index=True)

# No use adding to the dictionary if no grant numbers are given
epmc_data.dropna(subset=['WTgrants'], inplace=True)

# Split grants by comma
epmc_data['WTgrants'] = epmc_data['WTgrants'].apply(lambda x: x.split(','))
epmc_data['pmid'] = epmc_data['pmid'].astype(str)

pmid2grants = epmc_data.set_index('pmid').to_dict()['WTgrants']

with open('data/raw/EPMC/pmid2grants.json', 'w') as json_file:
    json.dump(pmid2grants, json_file)