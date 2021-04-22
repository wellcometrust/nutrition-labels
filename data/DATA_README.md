# Data folder

Notes about the data are given in more detail in the various docs, but here is a high level description of files and folders in the data folder.

You can download this data 2 ways:
1. If you have a Wellcome AWS account:
`make sync_data_from_s3`
or 
`sync_latest_files_from_s3` for just the essential files.

2. If you don't, you can download it (you will need to ask permission to access the drive though) from:
```
https://wellcomecloud.sharepoint.com/:f:/r/sites/DataLabs/Machine%20Learning/Nutrition%20Labels/data?csf=1&web=1&e=XHFn6n
```
this won't include data from the `raw/fortytwo/` and `processed/fortytwo/` folders since these contain private information and contain data for internal purposes.

- raw
    - EPMC/
    - ResearchFish/
    - expanded_tags/
    - fortytwo/
- prodigy
- processed
    - clustering/
    - ensemble/
    - evaluation/
    - fairness/
    - fortytwo/
    - model_test_results/
    - predictions/
    - top_health_datasets/
    - training_data/

## `raw/`

The raw folder contains downloaded data and manually tagged versions of this data for use in the training data.

360Giving grants data as downloaded July 2020:
- `wellcome-grants-awarded-2005-2019.csv`

A sample of 10 of the 360Giving grants data (useful for quick testing):
- `wellcome-grants-awarded-2005-2019_test_sample.csv`

Manually tagged versions:
- `wellcome-grants-awarded-2005-2019_manual_edit_relabeling.csv` (only used in testing human accuracy)
- `grants_sample_becky.csv` (only used in testing human accuracy)
- `wellcome-grants-awarded-2005-2019_manual_edit_Lizadditions.csv` (final version of tagged data)

##### `EPMC/`

Downloads from the EPMC API:
- `EPMC_relevant_tool_pubs.csv`
- `EPMC_relevant_pubs_query2.csv`

Manually tagged versions:
- `EPMC_relevant_tool_pubs_3082020.csv`
- `EPMC_relevant_pubs_query2_3082020.csv`

Due to Excel corruption the PMID-grants from both of these files are saved in
- `pmid2grants.json`

##### `ResearchFish/`

Downloads from ResearchFish data on fortytwo:
- `MedicalProductsAndInterventions.csv`
- `OtherOutputsAndKnowledge.csv`
- `ResearchDatabaseModels.csv`
- `ResearchToolsAndMethods.csv`
- `SoftwareAndTechnicalProducts.csv`

Manually tagged and collated together:
- `research_fish_manual_edit.csv`

##### `expanded_tags/`

After the expanded definition of 'tech' in 2021 the manually tagged datasets from EPMC (query 1 and 2), RF and grants data, were retagged and saved here:
- `EPMC_relevant_tool_pubs_3082020.csv`
- `EPMC_relevant_pubs_query2_3082020.csv`
- `research_fish_manual_edit.csv`
- `wellcome-grants-awarded-2005-2019_manual_edit_Lizadditions.csv`

##### `fortytwo/`

**This is a private folder, so the data is not intended to be available outside of Wellcome staff.**

It contains all the grants data (using the query: `SELECT Reference, Title, Synopsis FROM FortyTwo_Denormalised.[WTGT].[ApplicationsAndGrantDetails]`) pulled from fortytwo as of 20/04/2021:
- `all_grants_fortytwo_info_210420.csv`

Science tags for the comparison with the 'Data Modelling & Surveillance Grants' tag:
- `2020.10.09_DataModelling&SurveillanceGrants.csv`

## `prodigy/`

This folder contains several files from over the course of tagging grants data using Prodigy.

Datasets needed as input for the Prodigy tagging pipeline (specially Prodigy friendly formatted):
- `existing_training_data.jsonl`: The existing training data used for the 210129 Ensemble model
- `existing_test_data.jsonl`: The existing test data used for the 210129 Ensemble model
- `grants_data.jsonl`: All the grants data from from data/raw/wellcome-grants-awarded-2005-2019.csv

Prodigy databases/sets:
- `prodigy.db`
- `tech_grants/`
- `tech_grants_2/`

The final file to be used in training data is:
- `merged_tech_grants/merged_tech_grants.jsonl`

## `processed/`

##### `clustering/`
Clustering data after passing grants through `cluster_tech_grants.py`.

##### `ensemble/`
Grants data tagged with tech or not after passing through an ensemble model with `ensemble_model.py` or `ensemble_grant_tagger.py`.

##### `evaluation/`
Results of the model evaluation using `evaluate.py`.

##### `fairness/`
Results of the fairness evaluation via 'Fairness' notebooks.

##### `model_test_results/`
Results of model predictions using `predict_tech.py` (legacy).

##### `predictions/`
Results of model predictions using `predict.py`.

##### `top_health_datasets/`
Data for the analysis of the top health datasets found in the predicted tech grants.

##### `training_data/`
Training datasets created from `create_training_data.py`.

##### `fortytwo/`

**This is a private folder, so the data is not intended to be available outside of Wellcome staff.**

Using the grant IDs given in the 2021.03.08 training data I queried our grants warehouse (42) to find the original grants text data, this is here:
- `tech_210308_training_data_fortytwo_info.csv`

An older version of this file is:
- `tech_grantIDs_210126_training_data_fortytwo_info.csv`
