This document will describe the results of the tech grant model after some additional changes that were made to this project.

Firstly, the original training dataset has been expanded since the work done on this project in 2020.

# Training data 

## 2020 Training data

The training data used in 2020 was the `200807/training_data.csv` version. This consisted of 214 tech grants and 883 not tech grants. The process of tagging this and model results are described in `docs/Finding_Tech_Grants.md`.

## Expanding the definition of tech

We then expanded the definition of 'tech' and retagged some of the original training data. This process is described in `docs/Expanding_tech_grants.md`. In summary the changes were:

| Data type | Previous number | New number | Difference |
| --- | --- | --- | --- |
| RF tech data points | 23 | 144 | +122 |
| EPMC tech data points | 143 | 191 | +48 |
| Grants tech data points | 111 | 164 | +53 |
| Grants not-tech data points | 1004 | 358| -646 |

This resulted in the `210126/training_data.csv` training data with 347 tech grants and 349 not tech grants.

## Tagging training data using active learning in Prodigy

We then added to the training data using active learning. This process is described in `docs/Prodigy_training_data.md`. This created the outputted the `210221/training_data.csv` dataset which consists of 495 tech grants and 485 not tech grants.

## Final training data set

After some experimentation we realised that the ResearchFish and EPMC data points negatively effect the performance of the model. However they are useful in evaluating how well the model extends to find the 'hidden tech'. These experiments are discussed in `Training_data_sources.md`. After some refactoring and changing how the text was cleaned slightly the final training data set used it the `210308/training_data.csv` version.

A summary and comparison of the data changes is as follows:

| Tag code | Meaning | Number of grants - 200807 | Number of grants - 210126 | Number of grants - 210221 | Number of grants - 210308|
|---|---|--- |--- | --- | --- |
| 1 | Tech grants | 214 |347 | 495 | 313 |
| 0 | Not tech grants | 883 |349 | 485 | 488 |


We also outputted some evaluation data using just the ResearchFish and EPMC data respectively. This by running `python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.rf.ini` and `2021.03.29.epmc.ini` respectively.

# Training the models

The training data is then linked with text information about the grant. Usually this text comes from the publically available 360 giving dataset, and we include grant title, description and grant type. I experimented with using the 360 dataset vs data from grant tracker (42), and including the grant type or not, as well as cleaning the text in different ways.

## 42 data

Using the grant IDs given in the 2021.03.08 training data I queried our grants warehouse (42) to find the original grants text data. It's uncertain in what ways this data might be different to the 360 giving data, but it does appear to perform differently. Thus extra information about the training data is stored in `data/processed/fortytwo/tech_210308_training_data_fortytwo_info.csv`.

## Experiments

In `notebooks/Comparison of training experiments - March 2021.ipynb` I experiment with different model setups and whether the model's produced improve. Commonalities of these experiments are:
- Train/test on grants only data points (2021.03.08 training data)
- Same model parameters: relevant_sample_ratio = 1, test_size = 0.25, split_seed = 1, vectorizer_types = ['count', 'tfidf'], classifier_types = ['naive_bayes', 'SVM', 'log_reg']
- Evaluated on RF and EPMC data points (`2021.03.29.rf.ini` and `2021.03.29.epmc.ini` training data).
- Evaluated using 42 text data.

Experiments:

0. Baseline - `models/210316/`.
1. Remove stop words for count and tfidf vectorizers.
2. 1 + Don't train using grant types.
3. 1 + 2 + Apply clean_string to grants text.
4. 1 + Apply clean_string to grants text.
5. 1 + Apply clean_string to grants text + Train using fortytwo data (rather than usual 360 giving data).

There is quite a lot of variation in the results from the 6 different models (TFIDF + SVM, TFIDF + log_reg, ...). The trends remain the same regardless of experiment.

![](figures/training_experiments_all.png)

The average metrics of all models reveal that the best experiment is `remove stop words + clean strings + 42 training data` (experiment 5). This gives highest test recall and precision, and good EPMC evaluation metric.

![](figures/training_experiments_average.png)

It appears that the exclusion of the grant type data made the models perform worse.

|Metric|Experiment 0|Experiment 4|Experiment 5|
|---|---|---|---|
|Test precision (average of all models) | 0.791|0.822|0.825|
|Test recall (average of all models) | 0.829|0.833|0.848|
|Test precision (42 data) (average of all models) | 0.797|0.822|-|
|Test recall (42 data) (average of all models) | 0.867|0.844|-|
|EPMC accuracy (average of all models) | 0.676|0.706|0.699|
|RF accuracy (average of all models) | 0.514|0.524|0.510|

Note: When using the 42 data in the training the evaluation using 42 data is the same as the test metrics. Thus the 42 data evaluation is only really interesting to see how well the 360 giving data translates to 42 data.


## `grant_tagger.py` additions

Thus `grant_tagger.py` was adapted to:
- Remove stop words for count and tfidf vectorizers.
- Improved string cleaning of the training data.
- The random seed is set to 1. Earlier we tried to optimise the value picked for this, but then this would overfit to the test data - so this time it wasn't picked with any thought for optimisation.

A new model training config was made for training models (`configs/train_model/2021.03.31.ini`) which also takes the 42 grant data as an input to get the grant texts from.

## Performance

I ran:
```
python nutrition_labels/grant_tagger.py --config_path configs/train_model/2021.03.31.ini
```

I evaluated how well each model extended to make predictions of tech grants on the RF and EPMC datasets by running:
```
nutrition_labels/grant_tagger_evaluation.py --model_config configs/train_model/2021.03.31.ini --epmc_file_dir data/processed/training_data/210329epmc/training_data.csv --rf_file_dir data/processed/training_data/210329rf/training_data.csv
``` 
This script also outputs the test metrics for each model in one csv which gives:

| Date   | Vectorizer | Classifier  | f1    | precision_score | recall_score | EPMC accuracy | RF accuracy |
|--------|------------|-------------|-------|-----------------|--------------|---------------|-------------|
| 210331 | count      | naive_bayes | 0.837 | 0.74            | 0.962        | 0.791         | 0.557       |
| 210331 | count      | SVM         | 0.813 | 0.871           | 0.762        | 0.588         | 0.4         |
| 210331 | count      | log_reg     | 0.83  | 0.835           | 0.825        | 0.595         | 0.4         |
| 210331 | tfidf      | naive_bayes | 0.819 | 0.713           | 0.962        | 0.824         | 0.657       |
| 210331 | tfidf      | SVM         | 0.83  | 0.91            | 0.762        | 0.635         | 0.429       |
| 210331 | tfidf      | log_reg     | 0.844 | 0.878           | 0.812        | 0.709         | 0.457       |
| 210331 | bert       | naive_bayes | 0.802 | 0.77            | 0.838        | 0.818         | 0.571       |
| 210331 | bert       | SVM         | 0.866 | 0.845           | 0.888        | 0.608         | 0.443       |
| 210331 | bert       | log_reg     | 0.862 | 0.862           | 0.862        | 0.588         | 0.443       |
| 210331 | scibert    | naive_bayes | 0.845 | 0.807           | 0.888        | 0.466         | 0.243       |
| 210331 | scibert    | SVM         | 0.828 | 0.787           | 0.875        | 0.689         | 0.514       |
| 210331 | scibert    | log_reg     | 0.848 | 0.824           | 0.875        | 0.615         | 0.443       |

### Ensemble model


