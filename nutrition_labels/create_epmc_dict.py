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
from argparse import ArgumentParser

import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--epmc_query_one_path',
        help='Path to first query EPMC raw data csv',
        default='data/raw/EPMC/EPMC_relevant_tool_pubs.csv'
    )
    parser.add_argument(
        '--epmc_query_two_path',
        help='Path to second query EPMC raw data csv',
        default='data/raw/EPMC/EPMC_relevant_pubs_query2.csv'
    )
    parser.add_argument(
        '--output_path',
        help='Path for outputted PMID-grantID dict',
        default='data/raw/EPMC/pmid2grants.json'
    )
    args = parser.parse_args()

    # Original EPMC data from fortytwo (not manually labelled)
    epmc_query_one = pd.read_csv(args.epmc_query_one_path)
    epmc_query_two = pd.read_csv(args.epmc_query_two_path)

    epmc_data = pd.concat([epmc_query_one, epmc_query_two], ignore_index=True)

    # No use adding to the dictionary if no grant numbers are given
    epmc_data.dropna(subset=['WTgrants'], inplace=True)

    # Split grants by comma
    epmc_data['WTgrants'] = epmc_data['WTgrants'].apply(lambda x: x.split(','))
    epmc_data['pmid'] = epmc_data['pmid'].astype(str)

    pmid2grants = epmc_data.set_index('pmid').to_dict()['WTgrants']

    with open(args.output_path, 'w') as json_file:
        json.dump(pmid2grants, json_file)

