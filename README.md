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

## finding_most_common_cohort.py / finding_most_common_references.py 

To find the most common health data sets first run:

```
python nutrition_labels/finding_most_common_cohort.py
```

This uses data extracted from EPMC which has all the citations from all the papers produced from tech grants. 

This then produces a list of most cited papers which have been manually filtered to find only the health data set papers. 

To find the 5 most cited health data sets in the last five years run: 


```
python nutrition_labels/finding_most_common_references.py
```

This uses the EPMC client to find the number of citations each paper has and returns only the ones since 2015. 

The results are: 

-UK biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age.
-Cohort Profile: the 'children of the 90s'--the index offspring of the Avon Longitudinal Study of Parents and Children
-Data Resource Profile: Clinical Practice Research Datalink (CPRD).
-Cohort Profile: the Avon Longitudinal Study of Parents and Children: ALSPAC mothers cohort.
-Cohort profile: 1958 British birth cohort (National Child Development Study).
-Cohort Profile: the Whitehall II study.


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