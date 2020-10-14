# Data Nutrition Labels

Project description here 

## Set up the virtual environment

Running
```bash
make virtualenv
```
will create a virtual environment with the packages listed in `requirements.txt`.

Then when you want to develop and run code start this up by running
```bash
source build/virtualenv/bin/activate
```

## Download the data

Download the data for this project from 
```
https://wellcomecloud.sharepoint.com/:f:/r/sites/DataLabs/Machine%20Learning/Nutrition%20Labels/data?csf=1&web=1&e=XHFn6n
```
you will need to have been granted access to access this folder.

Make sure to upload any processed data to this folder too.

## Jupyter notebooks

To run a notebook run
```bash
source build/virtualenv/bin/activate
jupyter notebook
```

## grant_tagger.py

You can train and save a model to classify grants as being to do with tech or not (see definitions for this in `Finding_Relevant_Grants.md`) by running:

```
python nutrition_labels/grant_tagger.py --training_data_file data/processed/training_data.csv --vectorizer_type count --relevant_sample_ratio 1 --model_type naive_bayes --bert_type bert
```

where `vectorizer_type` can be 'count', 'tfidf' or 'bert', `model_type` can be 'naive_bayes', 'SVM' and 'log_reg', and `bert_type` (if using bert) can be 'bert' and 'scibert'.

## Getting the references of the tech grants' publications
Running

```
python nutrition_labels/finding_tech_grant_refs.py
```
will find all the PMIDs of publications which acknowledge any grant numbers in the tech grant we identified from the ensemble model (given in 'data/processed/ensemble_results.csv'). We found that 1138 of the 1790 tech grants (64%) were acknowledged in publications.

Then it uses the EPMC client to get all the references from each of these publications. This creates two useful outputs - data/processed/tech_grants_references_dict_2020-09-16.json information about each of the references found, and data/processed/tech_grants_reference_ids_counter_2020-09-16.json a counter of how many times each reference was found.

This data is then used to find the most common health data sets by first running:

```
python nutrition_labels/finding_most_common_references.py
```


This uses data extracted from EPMC ('data/processed/tech_grants_references_dict_2020-09-16.json' and 'data/processed/tech_grants_reference_ids_counter_2020-09-16.json') which has all the references from all the papers produced from tech grants.
The list of most referenced papers is then filtered to find ones only relating to tools using terms: 'cohort','profile','resource','biobank','Cohort','Profile','Biobank','Resource'. This is outputted in 'data/processed/health_data_sets_list.csv'.
This is then manually filtered to find only UK health data sets which can be found in 'data/processed/health_data_sets_list_manual_edit.csv' 

To find the 5 most cited health data sets in total and last five years run: 

```
python nutrition_labels/finding_most_common_cohort.py
```

This uses the EPMC client to find the number of citations each paper in the 'health_data_sets_list_manual_edit.csv' dataset above has and returns only the ones since 2015. 

The results are: 


| Study | Total citations| Recent citations |
|---|---|--- |
| UK biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age. | 814 | 814 |
| Cohort Profile: the 'children of the 90s'--the index offspring of the Avon Longitudinal Study of Parents and Children | 903 | 734 |
| Data Resource Profile: Clinical Practice Research Datalink (CPRD). | 664 | 664 |
| Cohort Profile: the Avon Longitudinal Study of Parents and Children: ALSPAC mothers cohort. | 609 | 518 |
| Cohort profile: 1958 British birth cohort (National Child Development Study). | 418 | 176 |
| Cohort Profile: the Whitehall II study. | 378 | 146 |


## Project structure

```
├── README.md
|
├── requirements.txt - a list of all the python packages needed for this project  
|
├── data
│   ├── processed - Data that we generate     
│   └── raw - Raw data                    
│
├── models            
│   Any models created
├── notebooks                
|   A place to put Jupyter notebooks
├── nutrition_labels
│   A place to put scripts
```