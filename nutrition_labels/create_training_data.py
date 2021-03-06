"""
Combine different sources of tagged data to create a training set.

Usage
----------
python nutrition_labels/create_training_data.py 
    --config_path configs/training_data/2021.03.08.ini

Description
----------

Create training data from any combination of 4 sources
- RF: Tech found from the researchfish data
- EPMC: Tech found from the researchfish data
- Grants: Tech or not tech found from the grants descriptions
- Prodigy grants : Same as above but using Prodigy active learning

Which of these sources to include is defined in the sources_include config parameter.

The pipeline for tagging training data was as follows:

Tag EPMC, tag ResearchFish, tag grants -> Merge together -> Input into Prodigy ->
    Tag other grants using active learning -> All Progidy input and output merged into one

Thus it's not super clear which are the grants tagged in Prodigy without looking
at the input. So even if you don't want to include certain data sources in the training
data, you still need to input them all to distinguish which ones were tagged in Prodigy.

If no Prodigy data is included (no prodigy_filedir in the config file) then this script
will still work (so old config files are still compatible).

In this script:
- Input all tagged data sources - EPMC, RF, Grants, Prodigy grants (optional)
- Clearly label which data came from which source (EPMC, RF, Grants (not using Prodigy), Prodigy grants)
- Filter out any sources not wanted
- Output this as the training data

"""

from argparse import ArgumentParser
from datetime import datetime
import configparser
import os
import json
import ast

import pandas as pd
import numpy as np

from nutrition_labels.grant_data_processing import (
    load_process_data_sources,
    merge_grants_sources,
    deduplicate_similar_grants,
)
from nutrition_labels.utils import clean_grants_data
from nutrition_labels.prodigy_training_data import load_prodigy_tags

SOURCES = ["RF", "EPMC", "Grants", "Prodigy grants"]


def output_counts(training_data):

    print("Number of training data points from the various sources:")
    multi_tag_data = training_data[pd.notnull(training_data[SOURCES]).sum(axis=1) != 1]
    print(f"Multiple sources: {len(multi_tag_data)}")
    single_tag_data = training_data[pd.notnull(training_data[SOURCES]).sum(axis=1) == 1]
    num_source = pd.notnull(single_tag_data[SOURCES]).sum(axis=0)
    print(f"Individual sources: \n{num_source}")


def combine_original_prodigy(training_data_original, prodigy_output):
    """
    Combine the original training data using EPMC, RF and grants
    with the data tagged in Prodigy, in a format which makes it clear
    which data point came from each source
    """

    original_data = pd.DataFrame(training_data_original.copy())
    original_data.drop_duplicates(subset="Internal ID", inplace=True)
    original_data.rename({"Relevance code": "Original tag"}, axis=1, inplace=True)
    original_data["RF"] = original_data["Normalised code - RF"].apply(
        lambda x: None if pd.isnull(x) else 1
    )
    original_data["EPMC"] = original_data["Normalised code - EPMC"].apply(
        lambda x: None if pd.isnull(x) else 1
    )
    original_data["Grants"] = original_data["Normalised code - grants"].apply(
        lambda x: None if pd.isnull(x) else (0 if int(x) == 5 else 1)
    )

    prodigy_data = pd.DataFrame.from_dict(
        prodigy_output, orient="index", columns=["Prodigy grants"]
    )
    prodigy_data["Internal ID"] = prodigy_data.index

    training_data = pd.merge(original_data, prodigy_data, how="outer", on="Internal ID")

    # If in original then set Prodigy grants to nan
    training_data.loc[
        pd.notnull(training_data["Original tag"]), "Prodigy grants"
    ] = np.nan

    return training_data[
        ["Internal ID", "Original tag", "RF", "EPMC", "Grants", "Prodigy grants"]
    ]


def process_filter_data_sources(
    grant_data, epmc_df, rf_df, grants_df, prodigy_output, sources_include
):

    grant_data = clean_grants_data(grant_data)

    # Training data without Prodigy
    training_data_original = merge_grants_sources(grant_data, epmc_df, rf_df, grants_df)

    # Combine with the Prodigy data (if any)
    training_data = combine_original_prodigy(training_data_original, prodigy_output)

    # Only include data with tags from the sources to include list
    training_data.dropna(subset=sources_include, how="all", inplace=True)

    # Get the tags for these from the columns in sources_include
    training_data["Relevance code"] = np.nan
    for source in sources_include:
        training_data["Relevance code"] = training_data["Relevance code"].fillna(
            training_data[source]
        )

    training_data = pd.merge(training_data, grant_data, how="left", on="Internal ID")
    training_data = deduplicate_similar_grants(training_data)

    return training_data


def create_training_data(config):

    # Load Prodigy data
    use_prodigy_directly = False
    if config["data"].get("prodigy_filedir"):
        prodigy_data = load_prodigy_tags(config["data"]["prodigy_filedir"])
        prodigy_output = {
            tag["Internal ID"]: tag["Relevance code"] for tag in prodigy_data
        }
        # Depending on the config file parameters, in some cases we can use the
        # Prodigy output directly for the training data
        if not config["data"].get("sources_include") or set(
            ast.literal_eval(config["data"]["sources_include"])
        ) == set(SOURCES):
            use_prodigy_directly = True
    else:
        prodigy_output = {}

    # Create training data
    if use_prodigy_directly:
        training_data = pd.DataFrame(
            list(prodigy_output.items()), columns=["Internal ID", "Relevance code"]
        )
        training_data.drop_duplicates(subset=["Internal ID"], inplace=True)
        training_data.reset_index()
    else:
        if not config["data"].get("sources_include"):
            sources_include = ["EPMC", "RF", "Grants"]
        else:
            sources_include = ast.literal_eval(config["data"]["sources_include"])

        if not set(sources_include).issubset(SOURCES):
            raise ValueError(
                f"The list sources_include in {args.config_path} needs to be a subset of {SOURCES}"
            )

        # Load datasets
        epmc_df, rf_df, grants_df = load_process_data_sources(
            config["data"]["epmc_tags_query_one_filedir"],
            config["data"]["epmc_tags_query_two_filedir"],
            config["data"]["rf_tags_filedir"],
            config["data"]["grant_tags_filedir"],
            ast.literal_eval(config["data_col_ranking"]["epmc_col_ranking"]),
            ast.literal_eval(config["data_col_ranking"]["grants_col_ranking"]),
            config["data"]["epmc_pmid2grants_dir"],
        )
        grant_data = pd.read_csv(config["data"]["grant_data_filedir"])

        # Process datasets - combine, remove sources
        training_data = process_filter_data_sources(
            grant_data, epmc_df, rf_df, grants_df, prodigy_output, sources_include
        )
        output_counts(training_data)
        training_data = training_data[["Internal ID", "Relevance code"]]

    print(f"Number tagged as 0: {sum(training_data['Relevance code']==0)}")
    print(f"Number tagged as 1: {sum(training_data['Relevance code']==1)}")

    # Output the data to a dated folder using the config version date
    # but convert this from 2020.08.07 -> 200807
    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]
    output_path = os.path.join("data/processed/training_data", config_version)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    training_data.to_csv(os.path.join(output_path, "training_data.csv"), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="configs/training_data/2021.03.08.ini",
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    create_training_data(config)
