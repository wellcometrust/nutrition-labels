# Tagging Additional Training Data Using Prodigy

## Pre-processing

Since we have already tagged a training dataset, we first need to process this data into the JSONL format useful for Prodigy.

`notebooks/Get prodigy format data.ipynb` creates three datasets:
1. `data/prodigy/existing_training_data.jsonl`: The existing training data used for the 210129 Ensemble model
2. `data/prodigy/existing_test_data.jsonl`: The existing test data used for the 210129 Ensemble model
3. `data/prodigy/grants_data.jsonl`: All the grants data from from data/raw/wellcome-grants-awarded-2005-2019.csv

In each case all the texts (title+description) have been cleaned, deduplicated and descriptions with 'Not available' have been removed.


## Using Prodigy

Prodigy has a feature for active learning whilst tagging training data. This way we could tag additional training data in an intelligent way - focussing on tagging grants which the model will learn most from, e.g. edge case grants, or ones dissimilar to others.

```
make prodigy_virtualenv
source build/prodigy_virtualenv/bin/activate
```

Create a database and input the existing training data split into it.
```
prodigy dataset tech_grants "Grants labelled with whether they are tech or not" --author Liz
prodigy db-in tech_grants data/prodigy/existing_training_data.jsonl
```

Tag new data points (from the list created in `notebooks/Get prodigy format data.ipynb`) whilst actively learning which data points should be tagged next in order to make maximum gains at training a logistic regression model (custom recipe in nutrition_labels/prodigy_textcat_teach.py).

```
prodigy textcat.teach-tech tech_grants data/prodigy/grants_data.jsonl -F nutrition_labels/prodigy_textcat_teach.py
```
This will give you a new score on the training and test data (in `data/prodigy/existing_test_data.jsonl`) every 10 annotations. The progress bar gives you the test F1 score.

## After tagging

When you have finished tagging save the dataset by running:
```
prodigy db-out tech_grants data/prodigy/tech_grants
```

Since we tag the text with accept or reject 'tech model' or accept or reject 'not tech model', it's important to remember that an accept=False result means the opposite tag is true.

The Prodigy data needed reformatting to a format suitable for training data. This can be created by running:
```
python nutrition_labels.prodigy_training_data.py --prodigy_data_dir 'data/prodigy/tech_grants/tech_grants.jsonl'
```
which will output the training data in a datestamped folder (e.g. 'data/processed/training_data/210210/training_data.csv').

Then, as usual, you use can this `grant_tagger.py` to train your model, e.g. by running:
```
python nutrition_labels/grant_tagger.py --training_data_file 'data/processed/training_data/210210/training_data.csv' --vectorizer_type count --model_type naive_bayes
```
with the different values for vectorizer_type, model_type and bert_type.
