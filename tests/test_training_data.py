import pytest
import tempfile
import os
import json

import pandas as pd

from nutrition_labels.grant_data_processing import (
    clean_grants_data,
    merge_grants_sources,
    deduplicate_similar_grants,
)
from nutrition_labels.prodigy_training_data import load_prodigy_tags
from nutrition_labels.create_training_data import process_filter_data_sources

uncleaned_grant_data = pd.DataFrame(
    [
        {
            "Description": '<p style="margin-left: 0px; margin-right: 0px"><strong>This</strong> has html.</p><p style="margin-left: 0px; margin-right: 0px">&nbsp;</p><ul style="list-style-type: disc"><li>It includes a list:</li><li>first in list,</li><li>second in list</li></ul>',
            "Internal ID": "111111/B/19/Z",
            "Award Date": "01/01/2000",
        },
        {
            "Description": "Grants description about a particular gene.",
            "Internal ID": "555555/A/20/Z",
            "Award Date": "02/01/2000",
        },
        {
            "Description": "Grantsdescriptionabout aparticular gene.",
            "Internal ID": "555555/Z/20/Z",
            "Award Date": "01/01/2000",
        },
        {
            "Description": "In this grant we hope to create new software",
            "Internal ID": "987654/A/19/Z",
            "Award Date": "01/01/2000",
        },
        {
            "Description": "This is a grant about the history of medicine",
            "Internal ID": "777777/A/19/Z",
            "Award Date": "01/01/2000",
        },
        {
            "Description": "This is a duplicated grant in the data",
            "Internal ID": "444444/Z/19/Z",
            "Award Date": "01/01/2000",
        },
        {
            "Description": "This is a duplicated grant in the data",
            "Internal ID": "444444/Z/19/Z",
            "Award Date": "01/01/2000",
        },
        {
            "Description": "Not available",
            "Internal ID": "123456/A/20/Z",
            "Award Date": "01/01/2000",
        },
        {
            "Description": None,
            "Internal ID": "222222/Z/19/Z",
            "Award Date": "01/01/2000",
        },
        {"Description": "", "Internal ID": "333333/Z/19/Z", "Award Date": "01/01/2000"},
    ]
)

epmc_df = pd.DataFrame(
    [
        {"Internal ID 6 digit": "111111", "Normalised code - EPMC": 2},
        {"Internal ID 6 digit": "555555", "Normalised code - EPMC": 1},
    ]
)

rf_df = pd.DataFrame(
    [
        {"Internal ID": "111111/B/19/Z", "Normalised code - RF": 3},
        {"Internal ID": "987654/A/19/Z", "Normalised code - RF": 1},
    ]
)

grants_df = pd.DataFrame(
    [
        {"Internal ID": "111111/B/19/Z", "Normalised code - grants": 5},
        {"Internal ID": "987654/A/19/Z", "Normalised code - grants": 1},
        {"Internal ID": "777777/A/19/Z", "Normalised code - grants": 5},
    ]
)


def test_clean_grants_data():

    grant_data = clean_grants_data(uncleaned_grant_data)

    html_grant_text = grant_data.iloc[0]["Description"]

    assert len(grant_data) == 6
    assert (
        html_grant_text
        == "This has html. It includes a list:first in list,second in list"
    )


def test_merge_grants_sources():

    grant_data = clean_grants_data(uncleaned_grant_data)

    merged_grant_data = merge_grants_sources(grant_data, epmc_df, rf_df, grants_df)

    ids_included = [
        "111111/B/19/Z",
        "987654/A/19/Z",
        "777777/A/19/Z",
        "555555/A/20/Z",
        "555555/Z/20/Z",
    ]

    # Check only tagged grant_data was kept in
    assert set(merged_grant_data["Internal ID"]) == set(ids_included)
    # Check grant with contradictory EPMC/grants tag was given EPMC tag
    assert (
        merged_grant_data.loc[merged_grant_data["Internal ID"] == "111111/B/19/Z"][
            "Relevance code"
        ][0]
        == 1
    )


def test_deduplicate_similar_grants():

    grant_data = pd.DataFrame(
        [
            {
                "Description": "Grants description about a particular gene.",
                "Internal ID 6 digit": "555555",
            },
            {
                "Description": "Grantsdescriptionabout aparticular gene.",
                "Internal ID 6 digit": "555555",
            },
        ]
    )
    grant_data = deduplicate_similar_grants(grant_data)

    assert len(grant_data) == 1


def test_load_prodigy_tags():

    prodigy_data = [
        {
            "id": "555555/A/20/Z",
            "text": "Grants description about a particular gene.",
            "label": "Tech grant",
            "answer": "ignore",
        },
        {
            "id": "987654/A/19/Z",
            "text": "In this grant we hope to create new software",
            "label": "Tech grant",
            "answer": "accept",
        },
        {
            "id": "777777/A/19/Z",
            "text": "This is a grant about the history of medicine",
            "label": "Tech grant",
            "answer": "reject",
        },
        {
            "id": "777778/A/19/Z",
            "text": "This is another grant about the history of medicine",
            "label": "Not tech grant",
            "answer": "accept",
        },
        {
            "id": "777779/A/19/Z",
            "text": "This is a tech grant",
            "label": "Not tech grant",
            "answer": "reject",
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        prodigy_data_dir = os.path.join(tmp_dir, "prodigy_data.jsonl")
        with open(prodigy_data_dir, "w") as json_file:
            for entry in prodigy_data:
                json.dump(entry, json_file)
                json_file.write("\n")
        training_data = load_prodigy_tags(prodigy_data_dir)

    correct_labels = {
        "987654/A/19/Z": 1,
        "777777/A/19/Z": 0,
        "777778/A/19/Z": 0,
        "777779/A/19/Z": 1,
    }
    assert len(training_data) == 4
    assert all(
        [correct_labels[t["Internal ID"]] == t["Relevance code"] for t in training_data]
    )


def test_process_filter_data_sources():

    made_up_data = []
    for i in range(1, 8):
        made_up_data.append(
            {
                "Description": "This is a grant description",
                "Internal ID": f"{str(i)*6}/A/19/Z",
                "Award Date": "01/01/2000",
            }
        )
    grant_data = pd.DataFrame(made_up_data)

    epmc_df = pd.DataFrame(
        [
            {"Internal ID 6 digit": "111111", "Normalised code - EPMC": 2},
            {"Internal ID 6 digit": "222222", "Normalised code - EPMC": 1},
        ]
    )

    rf_df = pd.DataFrame(
        [
            {"Internal ID": "111111/A/19/Z", "Normalised code - RF": 3},
            {"Internal ID": "333333/A/19/Z", "Normalised code - RF": 1},
        ]
    )

    grants_df = pd.DataFrame(
        [
            {"Internal ID": "111111/A/19/Z", "Normalised code - grants": 5},
            {"Internal ID": "444444/A/19/Z", "Normalised code - grants": 1},
            {"Internal ID": "555555/A/19/Z", "Normalised code - grants": 5},
        ]
    )

    prodigy_output = {
        "111111/A/19/Z": 1,
        "444444/A/19/Z": 0,
        "666666/A/19/Z": 0,
        "777777/A/19/Z": 1,
    }

    sources_include = ["EPMC", "RF", "Grants", "Prodigy grants"]
    training_data1 = process_filter_data_sources(
        grant_data, epmc_df, rf_df, grants_df, prodigy_output, sources_include
    )

    assert set(rf_df["Internal ID"]) < set(training_data1["Internal ID"])
    assert set(epmc_df["Internal ID 6 digit"]) < set(
        training_data1["Internal ID 6 digit"]
    )
    assert set(grants_df["Internal ID"]) < set(training_data1["Internal ID"])
    assert prodigy_output.keys() < set(training_data1["Internal ID"])
    assert training_data1["Prodigy grants"].notnull().sum() == 2

    sources_include = ["Grants", "Prodigy grants"]
    training_data2 = process_filter_data_sources(
        grant_data, epmc_df, rf_df, grants_df, prodigy_output, sources_include
    )

    assert set(["222222/A/19/Z", "333333/A/19/Z"]).isdisjoint(
        set(training_data2["Internal ID"])
    )
