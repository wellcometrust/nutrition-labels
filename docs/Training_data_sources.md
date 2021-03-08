
In the notebook `notebooks/Training data analysis.ipynb` I had a look at the extent and effect of the different sources of training data. I wanted to answer 3 broad questions:

1. How much training data comes from each of the 4 tagging sources:
- EPMC
- Research Fish
- Grants via Excel
- Grants via Prodigy active learning
2. How well does the 210223 ensemble do for each of these 4 sources?
3. Does training the models with no EPMC or RF training data contributions improve the model?

This analysis was performed on the following data:
1. The training data that went into Prodigy : `../data/processed/training_data/210126/training_data.csv`
2. The final Prodigy merged results (original training + new tags) : `../data/prodigy/merged_tech_grants/merged_tech_grants.jsonl`
3. Which grants were from the test set for the 210221 model runs: `../data/processed/model_test_results/test_data_210221.csv`
4. The predictions on all grants from the 210223 Ensemble model (which used the 210221 models): `../data/processed/ensemble/210223/wellcome-grants-awarded-2005-2019_tagged.csv`

# 1. How much training data comes from each of the 4 tagging sources?

The training data is labelled from:
- Multiple sources: 26
- RF: 57 (57 'tech')
- EPMC: 126 (126 'tech')
- Grants: 485 (138 'tech', 347 'not tech)
- Prodigy only: 286 (150 'tech', 136 'not tech)

Of the 26 multiple sources only 3 were not in agreement:
- 3 times the grant description was labelled 'not tech', but RF data labelled as 'tech'
- 0 times the grant description was labelled 'not tech', but EPMC data labelled as 'tech'
- 3 times the grant description and RF labels both said tech
- 13 times the grant description and EPMC labels both said tech
- 7 times both EPMC and RF labels said tech

# 2. How well does the 210223 ensemble do for each of these 4 sources?

Using the test data (i.e. not any data that went into training the models), the 210223 ensemble model performs differently for data points labelled from the different sources.

- Of the 19 ResearchFish labelled data points (all labelled as tech) **0.632** were correctly labelled as tech when using the grant description.
- Of the 35 EPMC labelled data points (all labelled as tech) **0.714** were correctly labelled as tech when using the grant description.

When tagging a grant as tech or not from the original grant descriptions the model performs better:
- 114 original grants via Excel: 'precision': 0.805, 'recall': **0.892**, 'f1': 0.846
- 73 grants tagged via Prodigy: 'precision': 0.771, 'recall': **0.902**, 'f1': 0.831
- 187 grants **either** tagged via Excel or Prodigy: 'precision': 0.787, 'recall': **0.897**, 'f1': 0.838

Ensemble performance on all test data (recap):
- 241 grants: 'precision': 0.849, 'recall': 0.811, 'f1': 0.829

# 3. Does training the models with no EPMC or RF training data contributions improve the model?

In the notebook I output `../data/prodigy/merged_tech_grants/merged_tech_grants_noepmcrf.jsonl` which is the Prodigy outputted data (`'../data/prodigy/merged_tech_grants/merged_tech_grants.jsonl'`) minus the additions from the EPMC or RF tagged training data (from the `training_data/210126/training_data.csv` training data).

This can be run in `prodigy_training_data.py` to create a new training data csv:
```
python -i nutrition_labels/prodigy_training_data.py --prodigy_data_dir 'data/prodigy/merged_tech_grants/merged_tech_grants_noepmcrf.jsonl'
```
which outputted `training_data/210305/training_data.csv`. 

| Tag code | Meaning | Number of grants - 210221 | Number of grants - 210305 |
|---|---|--- |--- |
| 1 | Relevant | 495 | 306 |
| 0 | Not relevant | 485 | 484 |

Then I start running models again:
```
python -i nutrition_labels/grant_tagger.py --training_data_file 'data/processed/training_data/210305/training_data.csv' --vectorizer_type tfidf --model_type naive_bayes
```
The split seed was set to 7 - but this time I didn't run any of the 'best seed' experiments - I just kept it set to what it was at before.

| Date | Vectorizer type | Model type | Bert type (if relevant) | Train F1 | Test F1 | Test precision | Test recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 210221 | count | log_reg | - | 1.000 | 0.772 | 0.806 | 0.741 |
| 210305 | count | log_reg | - | 1.000 | 0.805 | 0.831 | 0.780 |
| 210221 | count | naive_bayes | - | 1.000 | 0.774 | 0.820 | 0.732 |
| 210305 | count | naive_bayes | - | 1.000 | 0.836 | 0.779 | 0.902 |
| 210221 | count | SVM | - | 0.969 | 0.720 | 0.827 | 0.637 |
| 210305 | count | SVM | - | 0.977 | 0.748 | 0.795 | 0.707 |
| 210221 | tfidf | log_reg | - | 0.997 | 0.759 | 0.814 | 0.711 |
| 210305 | tfidf | log_reg | - | 1.000 | 0.829 | 0.90 | 0.768 |
| 210221 | tfidf | naive_bayes | - | 0.999 | 0.817 | 0.779 | 0.859 |
| 210305 | tfidf | naive_bayes | - | 1.000 | 0.825 | 0.768 | 0.890 |
| 210221 | tfidf | SVM | - | 1.000 | 0.736 | 0.846 | 0.652 |
| 210305 | tfidf | SVM | - | 1.000 | 0.778 | 0.903 | 0.683 |
| 210221 | bert | naive_bayes | bert | 0.730 | 0.746 | 0.803 | 0.696 |
| 210305 | bert | naive_bayes | bert | 0.762 | 0.775 | 0.795 | 0.756 |
| 210221 | bert | SVM | bert | 0.803 | 0.780 | 0.815 | 0.748 |
| 210305 | bert | SVM | bert | 0.866 | 0.848 | 0.843 | 0.854 |
| 210221 | bert | log_reg | bert | 0.993 | 0.789 | 0.817 | 0.763 |
| 210305 | bert | log_reg | bert | 1.000 | 0.888 | 0.862 | 0.915 |
| 210221 | bert | naive_bayes | scibert | 0.764 | 0.779 | 0.803 | 0.756 |
| 210305 | bert | naive_bayes | scibert | 0.823 | 0.795 | 0.810 | 0.780 |
| 210221 | bert | SVM | scibert | 0.793 | 0.783 | 0.839 | 0.733 |
| 210305 | bert | SVM | scibert | 0.835 | 0.832 | 0.848 | 0.817 |
| 210221 | bert | log_reg | scibert | 1.000 | 0.794 | 0.819 | 0.770 |
| 210305 | bert | log_reg | scibert | 1.000 | 0.875 | 0.897 | 0.854 |

Across most of these models all metrics increase - sometimes quite dramatically (a 0.1 increase on log_reg + BERT), despite a large decrease in the training data size.

