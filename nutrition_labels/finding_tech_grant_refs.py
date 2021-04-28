"""
This won't work without fortytwo credentials

From a lists of WT grants:
1. Get the PMIDs of publications linked to them
2. Find the list of references within each of these publications
3. Create a list of unique references that come up
4. Create a counter of how many times each reference has come up

Terminology:
'grant' -> acknowledged in a 'publication' -> 'references' of a publication
- a grant my be acknowledged in several publications
- a reference may be referenced in several publications
"""
from collections import defaultdict
import json
from datetime import date

import pandas as pd

from datalabs.epmc.client import EPMCClient
from datalabs.forty_two import forty_two

def yield_publications(forty_two_connection, query):
    yield from forty_two_connection.execute_query(query)

if __name__ == '__main__':
    
    today = date.today()

    grants = pd.read_csv('data/processed/ensemble/200907/ensemble_results.csv')
    grants = grants['Internal ID'].tolist()
    grants_6_digit = set([g[0:6] for g in grants])

    # Get the PMIDs of publications that acknowledge these grants using fortytwo

    forty_two_connection = forty_two.FortyTwo()

    query = f"""
            SELECT *
            FROM FortyTwo_Denormalised.[EPMC].[PublicationGrant]
            WHERE [grant WT 6 digit grantID] in {str(tuple(grants_6_digit))}
            """

    print(f"Finding publications that acknowledge {len(grants_6_digit)} grants...")
    pubs_per_grant = defaultdict(list)
    for i, pub_info in enumerate(yield_publications(forty_two_connection, query)):
        epmc_grant = pub_info['grant WT 6 digit grantID']
        if pub_info['pmid'].strip() != '':
            pubs_per_grant[epmc_grant].append((pub_info['pmid'], 'MED'))
        elif pub_info['pmcid'].strip() != '':
            pubs_per_grant[epmc_grant].append((pub_info['pmcid'], 'MED'))
        elif pub_info['doi'].strip() != '':
            pubs_per_grant[epmc_grant].append((pub_info['doi'], 'MED'))
        if i%1000 == 0:
            print(i)

    # 1138 of the 1790 tech grants (64%) are acknowledged in publications

    grants_with_pubs_perc = round(len(pubs_per_grant)*100/len(grants_6_digit))
    print(f'{len(pubs_per_grant)} of the {len(grants_6_digit)}',
        f'tech grants ({grants_with_pubs_perc}%) are acknowledged in publications')

    # Save

    with open(f'data/processed/tech_grants_pubs_per_grant_{today}.json', 'w') as fb:
        fb.write(json.dumps(pubs_per_grant))

    # Look to see if there are references for these pub IDs

    print(f"Finding references in the {len(pubs_per_grant)} publications ...")
    epmc_client = EPMCClient(max_retries=3)
    session = epmc_client.requests_session()

    reference_ids_counter = defaultdict(int)
    references_dict = defaultdict(list)
    # To keep track of how many publications do and don't have this information
    # Might be because they dont have any references, or no full text available
    has_references_pub = set()
    no_references_pub = set() 
    for i, (grant_id, pubs) in enumerate(pubs_per_grant.items()):
        if i%10 == 0:
            print(i)
        for pub_id, source in pubs:
            references = epmc_client.get_references(session, pub_id, source=source)
            if references:
                for reference in references:
                    ref_id = reference.get('id')
                    if ref_id:
                        references_dict[ref_id] = reference
                        reference_ids_counter[ref_id] += 1
                has_references_pub.add(pub_id)
            else:
                no_references_pub.add(pub_id)

    print(f'There were {len(references_dict)} unique references found within',
        f'the publications. {len(has_references_pub)} publications had references information',
        f"and {len(no_references_pub)} didn't.")

    # Save

    with open(f'data/processed/tech_grants_references_dict_{today}.json', 'w') as fb:
        output_list = [value for key, value in references_dict.items()]
        for i in output_list:
            if json.dumps(i)!='[]':
                json_ = json.dumps(i) + '\n'
                fb.write(json_)
    with open(f'data/processed/tech_grants_reference_ids_counter_{today}.json', 'w') as fb:
        fb.write(json.dumps(reference_ids_counter))
    with open(f'data/processed/tech_grants_has_references_pub_{today}.txt', 'w') as fb:
        for pubid in has_references_pub:
            fb.write(pubid + '\n')
    with open(f'data/processed/tech_grants_no_references_pub_{today}.txt', 'w') as fb:
        for pubid in no_references_pub:
            fb.write(pubid + '\n')
