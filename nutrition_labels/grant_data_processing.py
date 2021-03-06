import re
import configparser
from argparse import ArgumentParser
import ast
import os
import json

import pandas as pd
import numpy as np

from nutrition_labels.utils import clean_string, only_text, clean_grants_data


def merge_tags(data, person_cols_list):
    # person_cols_list: A list of the column names of people tagging
    #                       in order of preference.
    # e.g. If person [0] and person [1] have labelled the same row then use person [0]'s label
    data = data.dropna(subset=person_cols_list, how="all").reset_index(drop=True)

    merged_codes = []
    for _, row in data.iterrows():
        for person_col in person_cols_list:
            if pd.notnull(row[person_col]):
                merged_codes.append(row[person_col])
                break

    data["Merged code"] = merged_codes

    return data

def process_epmc(
    epmc_tags_query_one,
    epmc_tags_query_two,
    epmc_code_dict,
    col_ranking_list,
    pmid2grants,
):

    # Merge EPMC data and normalise the codes
    # Order of truth (if same row has been labelled): Becky > Nonie > Liz > Aoife
    epmc_tags = pd.concat([epmc_tags_query_one, epmc_tags_query_two], ignore_index=True)
    epmc_tags = merge_tags(epmc_tags, col_ranking_list)
    epmc_tags["Normalised code"] = [
        epmc_code_dict[str(int(code))] for code in epmc_tags["Merged code"]
    ]

    # No need to include tags if no grant number is given or you don't want to include
    # the grant in the training data
    epmc_tags.dropna(subset=["WTgrants", "Normalised code"], inplace=True)

    # Create list of dicts for each WT grant number given
    epmc_list = []
    for i, row in epmc_tags.iterrows():
        pmid = str(row["pmid"])  # Can sometimes be int
        grants = pmid2grants.get(pmid)
        for grant in grants:
            epmc_list.append(
                {
                    "pmid": pmid,
                    "Normalised code - EPMC": int(row["Normalised code"]),
                    "Internal ID 6 digit": grant,
                }
            )

    epmc_df = pd.DataFrame(epmc_list)
    epmc_df.drop_duplicates(
        subset=["Internal ID 6 digit", "Normalised code - EPMC"], inplace=True
    )

    return epmc_df


def process_RF(rf_tags, rf_code_dict):

    rf_tags.dropna(subset=["code "], inplace=True)
    rf_tags["Normalised code"] = [
        rf_code_dict[str(int(code))] for code in rf_tags["code "]
    ]

    # No need to include tags if no grant number is given or you don't want to include
    # the grant in the training data
    rf_tags.dropna(subset=["Grant Reference", "Normalised code"], inplace=True)

    # Create list of dicts for each WT grant number given
    rf_list = []
    for i, row in rf_tags.iterrows():
        # Strip whitespace from grant ref, since sometimes it has it
        rf_list.append(
            {
                "RF question": row["file"],
                "RF Name": row["Name"],
                "Normalised code - RF": int(row["Normalised code"]),
                "Internal ID": row["Grant Reference"].strip(),
            }
        )

    rf_df = pd.DataFrame(rf_list)
    rf_df.drop_duplicates(subset=["Internal ID", "Normalised code - RF"], inplace=True)

    return rf_df


def process_grants(grant_tags, grants_code_dict, col_ranking_list):

    # If Nonie and Liz have labelled it use Nonies
    grant_tags = merge_tags(grant_tags, col_ranking_list)
    grant_tags["Normalised code"] = [
        grants_code_dict[str(int(code))] for code in grant_tags["Merged code"]
    ]

    # No need to include tags if you don't want to include
    # the grant in the training data
    grant_tags.dropna(subset=["Normalised code"], inplace=True)

    # Create list of dicts for each WT grant number given
    grants_list = []
    for i, row in grant_tags.iterrows():
        # Strip whitespace from grant ref, since sometimes it has it
        grants_list.append(
            {
                "Normalised code - grants": int(row["Normalised code"]),
                "Internal ID": row["Internal ID"],
            }
        )
    grants_df = pd.DataFrame(grants_list)
    grants_df.drop_duplicates(
        subset=["Internal ID", "Normalised code - grants"], inplace=True
    )

    return grants_df


def load_process_data_sources(
    epmc_tags_query_one_filedir,
    epmc_tags_query_two_filedir,
    rf_tags_filedir,
    grant_tags_filedir,
    epmc_col_ranking,
    grants_col_ranking,
    epmc_pmid2grants_dir,
):
    """
    Load data from EPMC, RF and grants tags.
    Process each of these datasets into the form needed for merging together.
    Return processed versions of the  EPMC, RF and grants tags data.
    """

    epmc_tags_query_one = pd.read_csv(epmc_tags_query_one_filedir, encoding="latin")
    epmc_tags_query_two = pd.read_csv(epmc_tags_query_two_filedir)
    rf_tags = pd.read_csv(rf_tags_filedir)
    grant_tags = pd.read_csv(grant_tags_filedir)

    # Normalising the codes so they are all similar for different data sources
    # 'None' if you dont want to include these tags in the training data
    epmc_code_dict = {"1": 1, "2": 2, "3": 3, "4": None, "5": None, "6": None}
    rf_code_dict = {"1": 1, "2": 2, "3": 3, "4": None, "5": None}
    grants_code_dict = {"1": 1, "4": None, "5": 5}

    # Load pmid2grants dict for EPMC data
    with open(epmc_pmid2grants_dir) as f:
        pmid2grants = json.load(f)

    # Process each of the 3 data sources separately and output a
    # dataframe for each of grant numbers - cleaned tags links
    epmc_df = process_epmc(
        epmc_tags_query_one,
        epmc_tags_query_two,
        epmc_code_dict,
        epmc_col_ranking,
        pmid2grants,
    )
    rf_df = process_RF(rf_tags, rf_code_dict)
    grants_df = process_grants(grant_tags, grants_code_dict, grants_col_ranking)

    return epmc_df, rf_df, grants_df


def merge_grants_sources(grant_data, epmc_df, rf_df, grants_df, print_out=False):

    # Link with RF data (which uses 13 digit)
    grant_data = pd.merge(grant_data, rf_df, how="left", on="Internal ID")
    # Link with grant data (which uses 13 digit)
    grant_data = pd.merge(grant_data, grants_df, how="left", on="Internal ID")

    # Order grants by 1. those that have tags from RF or grants,
    # 2. if they havent been tagged yet, those with the most recent award date should go higher up
    grant_data.sort_values(
        by=["Normalised code - RF", "Normalised code - grants", "Award Date"],
        ascending=False,
        inplace=True,
    )

    # Link with EPMC data (which uses 6 digit)
    grant_data = pd.merge(grant_data, epmc_df, how="left", on="Internal ID 6 digit")

    if print_out:
        print(
            "Grants flagged as relevant in EPMC publications not found in grants data: "
        )
        print(
            set(epmc_df["Internal ID 6 digit"]).difference(
                set(grant_data["Internal ID 6 digit"])
            )
        )
        print(
            "PMIDs flagged as relevant in Wellcome EPMC publications but not found to link to any grants in the grants data: "
        )
        print(set(epmc_df["pmid"]).difference(set(grant_data["pmid"])))
        print("Grants flagged as relevant in RF data not found in grants data: ")
        print(set(rf_df["Internal ID"]).difference(set(grant_data["Internal ID"])))

    # Final list
    grant_data = grant_data.dropna(
        subset=[
            "Normalised code - RF",
            "Normalised code - grants",
            "Normalised code - EPMC",
        ],
        how="all",
    ).reset_index(drop=True)
    # Get the final single code for each grant:
    # If there is conflict in tags, the order of truth is:
    # 1. EPMC (this shows solid evidence of the output)
    # 2. RF (they might not have created the output at the time of the report)
    # 3. Grants (they might not have created the output at the time of the application)
    code = []
    for _, row in grant_data.iterrows():
        if pd.notnull(row["Normalised code - EPMC"]):
            code.append(row["Normalised code - EPMC"])
        elif pd.notnull(row["Normalised code - RF"]):
            code.append(row["Normalised code - RF"])
        elif pd.notnull(row["Normalised code - grants"]):
            code.append(row["Normalised code - grants"])
        else:
            print("Should have deduplicated this row?")

    # Map the final codes onto whether the grant is 'relevant' (1) or not (0)
    relevance_dict = {1: 1, 2: 1, 3: 1, 5: 0}
    grant_data["Relevance code"] = [relevance_dict[int(c)] for c in code]

    return grant_data


def deduplicate_similar_grants(
    grant_data, text_column="Description", grant_id_column="Internal ID 6 digit"
):
    """
    Some descriptions have some spaces deleted. We don't want to include
    grants from the same family (6 digit ID) if they have effectively the same description
    """
    grant_data = grant_data.copy()
    grant_data["Temporary column"] = grant_data[text_column].str.replace(r"\W", "")
    grant_data = grant_data.drop_duplicates([grant_id_column, "Temporary column"])
    grant_data.drop("Temporary column", axis=1, inplace=True)

    return grant_data


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="configs/training_data/2021.01.26.ini",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    epmc_df, rf_df, grants_df = load_process_data_sources(
        config["data"]["epmc_tags_query_one_filedir"],
        config["data"]["epmc_tags_query_two_filedir"],
        config["data"]["rf_tags_filedir"],
        config["data"]["grant_tags_filedir"],
        ast.literal_eval(config["data_col_ranking"]["epmc_col_ranking"]),
        ast.literal_eval(config["data_col_ranking"]["grants_col_ranking"]),
        config["data"]["epmc_pmid2grants_dir"],
    )

    print(f"Tagged data to include from EPMC: {len(epmc_df)}")
    print(f"Tagged data to include from RF: {len(rf_df)}")
    print(f"Tagged data to include from grant descriptions: {len(grants_df)}")

    grant_data = pd.read_csv(config["data"]["grant_data_filedir"])
    grant_data = clean_grants_data(grant_data)

    grant_data = merge_grants_sources(
        grant_data, epmc_df, rf_df, grants_df, print_out=True
    )

    grant_data = deduplicate_similar_grants(grant_data)
    # Note: This deduplication is after the general cleaning since it deduplicates on
    # 6 digit number, which needs to come after the merging with RF/Grants data

    num_irrelevant = len([i for i in grant_data["Relevance code"] if i == 0])
    print(f"Number tagged as 0: {num_irrelevant}")
    print(f"Number tagged as 1: {len(grant_data) - num_irrelevant}")
    grant_data = grant_data[
        [
            "Internal ID",
            "RF question",
            "RF Name",
            "pmid",
            "Relevance code",
            "Normalised code - RF",
            "Normalised code - grants",
            "Normalised code - EPMC",
            "Description",
            "Title",
            "Grant Programme:Title",
        ]
    ]

    # Output the data to a dated folder using the config version date
    # but convert this from 2020.08.07 -> 200807
    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]
    output_path = os.path.join("data/processed/training_data", config_version)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    grant_data.to_csv(os.path.join(output_path, "training_data.csv"), index=False)
