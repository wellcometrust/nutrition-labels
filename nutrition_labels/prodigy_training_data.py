"""
Create training data file from output of Prodigy tagging,
note that this dataset includes the output from grant_data_processing.py
"""

from argparse import ArgumentParser
from datetime import datetime
import os
import json

import pandas as pd


def load_prodigy_tags(prodigy_data_dir):

    cat2bin = {"Not tech grant": 0, "Tech grant": 1}

    training_data = []
    with open(prodigy_data_dir, "r") as json_file:
        for json_str in list(json_file):
            data = json.loads(json_str)
            if data["answer"] != "ignore":
                annotation = {"Internal ID": data["id"], "Grant texts": data["text"]}
                label = cat2bin[data["label"]]
                if data["answer"] == "accept":
                    annotation["Relevance code"] = label
                else:
                    # If label=1, append 0
                    # if label=0, append 1
                    annotation["Relevance code"] = abs(label - 1)
                training_data.append(annotation)

    return training_data


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--prodigy_data_dir",
        help="Path to prodigy output dataset file",
        default="data/prodigy/tech_grants/tech_grants.jsonl",
    )
    args = parser.parse_args()

    datestamp = datetime.now().date().strftime("%y%m%d")

    training_data = load_prodigy_tags(args.prodigy_data_dir)

    training_data = pd.DataFrame(training_data)
    training_data.drop_duplicates(subset=["Internal ID"], inplace=True)
    training_data.reset_index()

    output_path = os.path.join("data/processed/training_data", datestamp)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    training_data.to_csv(os.path.join(output_path, "training_data.csv"), index=False)
