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

Create a database and input the existing training data split (520 data points) into it.
```
prodigy dataset tech_grants_2 "Grants labelled with whether they are tech or not" --author Liz
prodigy db-in tech_grants_2 data/prodigy/existing_training_data.jsonl
```

Tag new data points (from the list created in `notebooks/Get prodigy format data.ipynb`) whilst actively learning which data points should be tagged next in order to make maximum gains at training a logistic regression model (custom recipe in nutrition_labels/prodigy_textcat_teach.py).
```
prodigy textcat.teach-tech tech_grants_2 data/prodigy/grants_data.jsonl --label 'Tech grant' -s "Low scoring tech grant predictions" -p "prefer_uncertain" -F nutrition_labels/prodigy_textcat_teach.py
```
This will give you a new score on the training and test data (in `data/prodigy/existing_test_data.jsonl`) every 10 annotations. The progress bar gives you the test F1 score - calculated on a hold out test dataset from `data/prodigy/existing_test_data.jsonl`, which wasn't imported as training data.

The optional "session_name" parameter can be changed with a useful tag for this bit of tagging, if not populated it will be filled with the date and time you started the session.

The optional "sorter" parameter is where you choose which Prodigy sorting method (see [here](https://prodi.gy/docs/api-components#sorters)) to use, by default this is `prefer_uncertain`.

## Tagging additions

All in the dataset `tech_grants_2`:

|session_name | Description | Number not ignored | 
|---|---|---|
| Originally labelled training data | Without using Prodigy | 520 |
| Low scoring tech grant predictions | Using Prodigy with `prefer_uncertain` sorting | 100 |
| High scoring tech grant predictions | Using Prodigy with `prefer_high_scores` sorting | 200 |


## After tagging

When you have finished tagging save the dataset by running:
```
prodigy db-out tech_grants_2 data/prodigy/tech_grants_2
```

Then add the hold-out test dataset to it with:
```
prodigy dataset test_data "210129 Ensemble model test dataset" --author Liz
prodigy db-in test_data data/prodigy/existing_test_data.jsonl
prodigy db-merge tech_grants_2,test_data merged_tech_grants
prodigy db-out merged_tech_grants data/prodigy/merged_tech_grants
```

Since we tag the text with accept or reject 'tech model' or accept or reject 'not tech model', it's important to remember that an accept=False result means the opposite tag is true.

The Prodigy data needed reformatting to a format suitable for training data. This can be created by running:
```
python nutrition_labels/prodigy_training_data.py --prodigy_data_dir 'data/prodigy/merged_tech_grants/merged_tech_grants.jsonl'
```
which will output the training data in a datestamped folder (e.g. 'data/processed/training_data/210210/training_data.csv').

Then, as usual, you use can this `grant_tagger.py` to train your model, e.g. by running:
```
python nutrition_labels/grant_tagger.py --training_data_file 'data/processed/training_data/210210/training_data.csv' --vectorizer_type count --model_type naive_bayes
```
with the different values for vectorizer_type, model_type and bert_type.

## 210218 training data results

| Tag code | Meaning | Number of grants |
|---|---|--- |
| 1 | Relevant | 592 |
| 0 | Not relevant | 576 |

### Variability in the results:

I reran `grant_tagger_seed_experiments.py` with the training data in `data/processed/training_data/210218/training_data.csv`. This script trains several models with different random seeds and teamed with the `Seed variability.ipynb` notebook we can pick a new 'best' random seed to choose our training/test split for all models.

New results:

| Model | Mean/std/range test accuracy | Mean/std/range test f1 | Mean/std/range test precision_score | Mean/std/range test recall_score |
| ----- | ------------------ | ------------ | ------------------------- | ---------------------- |
|bert_SVM_bert_210218|0.772/0.024/(0.729, 0.805)|0.775/0.026/(0.739, 0.818)|0.772/0.027/(0.724, 0.804)|0.78/0.039/(0.742, 0.861)|
|bert_SVM_scibert_210218|0.782/0.025/(0.736, 0.818)|0.784/0.028/(0.739, 0.817)|0.788/0.03/(0.741, 0.849)|0.781/0.041/(0.722, 0.854)|
|bert_log_reg_bert_210218|0.81/0.024/(0.781, 0.849)|0.814/0.028/(0.779, 0.866)|0.803/0.027/(0.765, 0.845)|0.827/0.047/(0.762, 0.918)|
|bert_log_reg_scibert_210218|0.824/0.033/(0.781, 0.866)|0.826/0.034/(0.775, 0.863)|0.825/0.027/(0.786, 0.872)|0.828/0.047/(0.762, 0.887)|
|bert_naive_bayes_bert_210218|0.744/0.024/(0.702, 0.781)|0.749/0.023/(0.715, 0.781)|0.745/0.019/(0.708, 0.77)|0.754/0.034/(0.697, 0.806)|
|bert_naive_bayes_scibert_210218|0.773/0.016/(0.75, 0.798)|0.779/0.014/(0.754, 0.8)|0.767/0.022/(0.732, 0.814)|0.792/0.024/(0.754, 0.833)|
|count_SVM_210218|0.78/0.027/(0.736, 0.822)|0.781/0.023/(0.755, 0.821)|0.789/0.039/(0.732, 0.877)|0.776/0.041/(0.735, 0.854)|
|count_log_reg_210218|0.8/0.022/(0.767, 0.842)|0.804/0.021/(0.764, 0.84)|0.802/0.031/(0.772, 0.877)|0.808/0.04/(0.728, 0.861)|
|count_naive_bayes_210218|0.812/0.023/(0.781, 0.849)|0.827/0.021/(0.797, 0.86)|0.774/0.02/(0.751, 0.823)|0.89/0.032/(0.833, 0.937)|
|tfidf_SVM_210218|0.814/0.019/(0.788, 0.842)|0.817/0.02/(0.788, 0.841)|0.817/0.029/(0.788, 0.888)|0.818/0.042/(0.762, 0.903)|
|tfidf_log_reg_210218|0.808/0.019/(0.788, 0.839)|0.812/0.017/(0.792, 0.836)|0.807/0.031/(0.771, 0.881)|0.819/0.034/(0.781, 0.882)|
|tfidf_naive_bayes_210218|0.767/0.028/(0.726, 0.815)|0.805/0.022/(0.775, 0.839)|0.702/0.037/(0.651, 0.758)|0.945/0.021/(0.901, 0.979)|


Comparison to the results before Prodigy data additions:

| Model | Mean test f1 - 210126 | Mean test f1 - 210218 |  Mean test precision_score - 210126|  Mean test precision_score - 210218| Mean test recall_score - 210126| Mean test recall_score - 210218|
| ----- | ------------------ | ------------ | ------------------------- | ---------------------- |------------------------- | ---------------------- |
| count_naive_bayes|0.809|0.827|0.755|0.774|0.875|0.890|
| count_log_reg|0.769|0.804|0.793|0.802 |0.751|0.808 |
| count_SVM|0.711| 0.781|0.767|0.789|0.670|0.776  |
| tfidf_naive_bayes|0.784| 0.805 |0.714|0.702 |0.886|0.945 |
| tfidf_log_reg|0.768|0.812 |0.819|0.807|0.736|0.819  |
| tfidf_SVM|0.748|0.817|0.836|0.817|0.688|0.818 |
| bert_naive_bayes_bert|0.738|0.749|0.723|0.745|0.757|0.754 |
| bert_log_reg_bert|0.768|0.814|0.770|0.803|0.769|0.827|
| bert_SVM_bert|0.780|0.775|0.773|0.772|0.794|0.78|
| bert_naive_bayes_scibert   |  0.789|0.779| 0.772  |0.767|0.812|0.792|
| bert_SVM_scibert|0.782|0.784|0.810|0.788|0.764|0.781|
| bert_log_reg_scibert|0.798|0.826|0.807|0.825|0.791|0.828|

### Best seed:
We calculated the highest average metrics over all models for the different random seeds used in `Seed variability.ipynb`.

The seed 7 gives good results like last time.

### Rerunning all models

We ran:

```
python nutrition_labels/grant_tagger.py --training_data_file 'data/processed/training_data/210218/training_data.csv' --vectorizer_type count --model_type naive_bayes --bert_type scibert
```
with the different values for vectorizer_type, model_type and bert_type.

A comparison with the previous results (201022 - old definition, 210128 - new definition but no additional Prodigy data):

| Date | Vectorizer type | Model type | Bert type (if relevant) | Train F1 | Test F1 | Test precision | Test recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 201022 | count | log_reg | - | 1.000 | 0.842 | 0.814 | 0.873 |
| 210128 | count | log_reg | - | 1.000 | 0.795 | 0.787 | 0.805 |
| 210218 | count | log_reg | - | 0.998| 0.840 | 0.877 | 0.807 |
| 201022 | count | naive_bayes | - | 1.000 | 0.864 | 0.810 | 0.927 |
| 210128 | count | naive_bayes | - | 1.000 | 0.827 | 0.743 | 0.931 |
| 210218 | count | naive_bayes | - | 0.998 | 0.860 | 0.823 | 0.9 |
| 201022 | count | SVM | - | 0.994 | 0.847 | 0.839 | 0.855 |
| 210128 | count | SVM | - | 0.981 | 0.786 | 0.791 | 0.782 |
| 210218 | count | SVM | - | 0.981 | 0.814 | 0.877 | 0.76 |
| 201022 | tfidf | log_reg | - | 1.000 | 0.844 | 0.852 | 0.836 |
| 210128 | tfidf | log_reg | - | 0.996 | 0.849 | 0.859 | 0.839 |
| 210218 | tfidf | log_reg | - | 0.996| 0.835 | 0.881 | 0.793 |
| 201022 | tfidf | naive_bayes | - | 1.000 | 0.846 | 0.765 | 0.945 |
| 210128 | tfidf | naive_bayes | - | 1.000 | 0.830 | 0.735 | 0.954 |
| 210218 | tfidf | naive_bayes | - | 0.988 | |0.839 | 0.758 | 0.94 |
| 201022 | tfidf | SVM | - | 1.000 | 0.822 | 0.846 | 0.800 |
| 210128 | tfidf | SVM | - | 1.000 | 0.828 | 0.854 | 0.805 |
| 210218 | tfidf | SVM | - |0.998| 0.838 | 0.888 | 0.793 |
| 201022 | bert | naive_bayes | bert | 0.713 | 0.757 | 0.813 | 0.709 |
| 210128 | bert | naive_bayes | bert | 0.748 | 0.789 | 0.745 | 0.839 |
| 210218 | bert | naive_bayes | bert | 0.730| 0.744 | 0.762 | 0.727 |
| 201022 | bert | SVM | bert | 0.819 | 0.881 | 0.825 | 0.945 |
| 210128 | bert | SVM | bert | 0.822 | 0.809 | 0.752 | 0.874 |
| 210218 | bert | SVM | bert | 0.815| 0.785 | 0.804 | 0.767 |
| 201022 | bert | log_reg | bert | 1.000 | 0.825 | 0.797 | 0.855 |
| 210128 | bert | log_reg | bert | 1.000 | 0.775 | 0.758 | 0.793 |
| 210218 | bert | log_reg | bert | 0.992| 0.824 | 0.822 | 0.827 |
| 201022 | bert | naive_bayes | scibert | 0.772 | 0.796 | 0.811 | 0.782 |
| 210128 | bert | naive_bayes | scibert | 0.816 | 0.842 | 0.802 | 0.885 |
| 210218 | bert | naive_bayes | scibert | 0.776| 0.800 | 0.814 | 0.787 |
| 201022 | bert | SVM | scibert | 0.776 | 0.879 | 0.904 | 0.855 |
| 210128 | bert | SVM | scibert | 0.831 | 0.851 | 0.819 | 0.885 |
| 210218 | bert | SVM | scibert | 0.813| 0.817 | 0.849 | 0.787 |
| 201022 | bert | log_reg | scibert | 1.000 | 0.814 | 0.762 | 0.873 |
| 210128 | bert | log_reg | scibert | 1.000 | 0.775 | 0.758 | 0.793 |
| 210218 | bert | log_reg | scibert | 0.997| 0.853 | 0.853 | 0.853 |

### Ensemble model

I ran:
```
python nutrition_labels/ensemble_model.py
```
with:
- split_seed = 7 (in grant_tagger.py)
- Training data from 210218
- F1 >= 0.8
- Precision >= 0.8
- Recall >= 0.8
- Models trained after = 210218
- Models trained before = 210218

