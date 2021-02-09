# Tagging Training Data

Prodigy has a feature for active learning whilst tagging training data. This way we could tag additional training data in an intelligent way - focussing on tagging grants which the model will learn most from, e.g. edge case grants, or ones dissimilar to others.

```
make prodigy_virtualenv
source build/prodigy_virtualenv/bin/activate
```

Create a database and input the existing training data split into it. This format of the data was created in `notebooks/Get prodigy format data.ipynb`.
```
prodigy dataset tech_grants "Grants labelled with whether they are tech or not" --author Liz
prodigy db-in tech_grants data/prodigy/existing_training_data.jsonl
```

Tag new data points (from a list created in `notebooks/Get prodigy format data.ipynb`) whilst actively learning which data points should be tagged next in order to make maximum gains at training a logistic regression model (custom recipe in nutrition_labels/prodigy_textcat_teach.py).

```
prodigy textcat.teach-tech tech_grants en_core_web_sm data/prodigy/grants_data.jsonl -F nutrition_labels/prodigy_textcat_teach.py --label 'Tech grant','Not tech grant'
prodigy db-out tech_grants data/prodigy/tech_grants
```
This will give you a new score on the training and test data (in `data/prodigy/existing_test_data.jsonl`) every 10 annotations. The progress bar gives you the test score.


Since we tag the text with accept or reject 'tech model' or accept or reject 'not tech model', it's important to remember that an accept=False result means the opposite tag is true.