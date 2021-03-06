# Finding Tech Grants

The first step in the Nutrition Labels project is to classify Wellcome grants as producing datasets/models/tools. Since there are 17,000 grants in the publically avaiable grants data from 2005 to 2019 this would be very time consuming to do by manually reading all the grants descriptions. Furthermore the grants description may not be enough to tell you whether the grant produced such an outcome.

Thus, we will create a model to predict whether a grant was likely to contribute to a dataset or tool ('relevant') or not. The initial step to this is to create a training data set of grants we know have and haven't produced relevant outcomes.

## Relevant grants

### Definitions

- Tool: Any peice of code or script that is used on a medical data set to process it for further analysis. Tasks that this includes can be, cleaning data, linking two data sets, annotating data or a platform or web resource for data
- Model: Any machine learning or statistical model that can be translated to a clinical enviroment, not to offer further insight for scientific research
- Medical Data Set: a data set containing at least in part medical data such as genetic data linked with a specific disease, electronic health records or imaging data

### Assumptions

- Tools or models that created a normal or healthy model of medical data such as a healthy (human) MRI scan or ECG were defined as tool
- Tools that annotated genetic regions were not included as tools unless they were linking them specifically to a human disease
- Code lists for determining patients with diseases in electronic health records were labeled as tools
- In tagging publication abstracts: mention of producing a particularly novel piece of software/model but only broadly related to humans (e.g. genetics), not a specific human disease, is still ok to include as a tool or model
- In tagging publication abstracts: a publication specifically about clinical trial results will often mention data collection, but we won't tag this as being about a dataset since it was created as a byproduct of investigating something else, not an end in itself


## Raw data

There were three possible data sources we looked at when tagging whether a grant was relevant or not.

1. Grant descriptions
2. ResearchFish questionnaire data
3. EPMC publication abstracts from publications with Wellcome funding acknowledged

### 1. Grant descriptions

The first step was reading the grant descriptions and trying to discern if they were relevant or not. It became clear that a really small proportion were relevant. The grants were filtered first to speed up this process, the filters applied were:

The grant descriptions were filtered if the following key words were found in the grant description:
['platform','software','library','mining','web resouce','data management','pipeline','tool','biobank','health data','medical records']

It should be noted that when searching for these words there was no space put before mining so it returned grant descriptions with words like determining etc.

873 of these grants were tagged as
- 1 = Relevant
- 5 = Not Relevant

This tagged data is in `data/raw/wellcome-grants-awarded-2005-2019_manual_edit_Lizadditions.csv`.

### 2. ResearchFish questionnaire data

Using the ResearchFish platform we've asked our grantholders over the last couple of years to self-report outcomes of their grants. There are a few fields for this which might flag which grants have outputted datasets/tools, these are in fortytwo (`FortyTwo_Denormalised.[ResearchFish]`) and the table names are:
- Medical Products, Interventions & Clinical Trials (fortytwo table: `MedicalProductsAndInterventions`)
- Research Databases & Models (fortytwo table: `ResearchDatabaseModels`)
- Research Tools & Methods (fortytwo table: `ResearchToolsAndMethods`)
- Software & Technical Products (fortytwo table: `SoftwareAndTechnicalProducts`)
- Other Outputs & Knowledge/Future Steps (fortytwo table: `OtherOutputsAndKnowledge`)

This yielded 1264 outcome reports, all these outcome descriptions were read and tagged as being
- 1 = Relevant tool
- 2 = Relevant model
- 3 = Relevant data set
- 4 = More information needed
- 5 = Not relevant

This tagged data is in `data/raw/ResearchFish/research_fish_manual_edit.csv`.

### 3. EPMC publication abstracts

#### Query 1
A final source to find relevant grants was to look for certain keywords in Wellcome acknowledged EPMC publication abstracts. This data was again from fortytwo (`FortyTwo_Denormalised.[EPMC].[PublicationFull]`).

The list of keywords used to filter this data is:
["platform", "software", "web resource", "pipeline", "toolbox", "data linkage", "database linkage", "record linkage", "linkage algorithm", "python package", "r module", "python script", "web tool", "web based tool"]


This yielded 3129 publications outputted in `EPMC_relevant_tool_pubs.csv`. 619 of these were tagged as being

- 1 = Relevant tool
- 2 = Relevant model
- 3 = Relevant data set
- 4 = More information needed
- 5 = Not relevant
- 6 = Edge case

This tagged data is in `data/raw/EPMC/EPMC_relevant_tool_pubs_3082020.csv`.

#### Query 2

After tagging some of the above grants, we decided to do another query of the EPMC data with the lists:
```
list1 = ['machine learning', 'model', 'data driven', 'web tool', 'web platform']
list2 = ['diagnosis', 'disease', 'clinical', 'drug discovery']
list3 = [
    'electronic health records', 'electronic medical records',
    'electronic patient records', 'biobank'
    ]
cap_list = ['EHR', 'EMR']
```
We would include a publication if the abstract contained words from `list1` and `list2`, or anything from `list3`, or anything from list `cap_list` (which is case sensitive).
We didn't include publications already found from the first query.

This yielded 9709 publications outputted in `EPMC_relevant_pubs_query2.csv`.

#### Excel EPMC grants list corruption

The EPMC data has a comma separated list of grant numbers associated with each pmid. During the tagging of the EPMC data using Excel this list is corrupted - Excel converts it to one number and will round to 15 significant figures - therefore a list of more than 2 6-digit grant numbers will be corrupted. For example, "086151,104104,196287" becomes "86,151,104,104,196,200". Therefore we process the raw EPMC data into a dictionary of `pmid: list of grant numbers`. This dictionary can be used to get the grant numbers rather than using the corrupted numbers in the tagged EPMC csvs. This is done by running:
 
```
python nutrition_labels/create_epmc_dict.py --epmc_query_one_path data/raw/EPMC/EPMC_relevant_tool_pubs.csv --epmc_query_two_path data/raw/EPMC/EPMC_relevant_pubs_query2.csv --output_path data/raw/EPMC/pmid2grants.json
```

## Compiling the training data

By running `python nutrition_labels/grant_data_processing.py` the three sets of tagged training data are combined with the publically available grants data `data/raw/wellcome-grants-awarded-2005-2019.csv`. This is outputted in `data/processed/training_data.csv`.

Not all the tagged data from ResearchFish (5 grants) or the EPMC publications (7 grants) were able to be linked back to the grants data since the grant reference wasn't found. Also for the EPMC data often only the 6 digit grant reference was given, in these cases we decided to combine this with the most recent grant in the grants data.

This dataset comprises of a grant reference, the grant title and description and also the tagged code.

### Training data summary - 30th June 2020

| Tag code | Meaning | Number of grants |
|---|---|--- |
| 1 | Relevant tool | 41 |
| 2 | Relevant model | 14 |
| 3 | Relevant dataset | 13 |
| 5 | Not relevant | 791 |

### Training data summary - 15th July 2020

| Tag code | Meaning | Number of grants |
|---|---|--- |
| 1 | Relevant | 292 |
| 0 | Not relevant | 989 |

### Training data summary - 7th August 2020

| Tag code | Meaning | Number of grants |
|---|---|--- |
| 1 | Relevant | 214 |
| 0 | Not relevant | 883 |

## Human accuracy

Using the EPMC data from 3082020 which includes tags from Nonie, Liz, Becky and Aoife (the data that went into the 7th August training data), and using the grants tagged data from 14th July, and some extra grants data Becky labelled and Nonie re-labelled, we ran:
```
python nutrition_labels.human_accuracy.py
```
to compare the tagging amongst different people.

We compare our tagging on EMPC data and the grants data separately.

### Evaluation between Nonie and Liz's EPMC tags

Proportion of times we exactly agree on tool, dataset, model, not relevant: 0.76 out of 51 we both labelled.

|             |Liz tag 5  |Liz tag 1  |Liz tag 2  |Liz tag 3|
|---|---|---|---|---|
|Nonie tag 5         |30          |2          |5          |0|
|Nonie tag 1          |1          |9          |1          |0|
|Nonie tag 2          |1          |0          |0          |0|
|Nonie tag 3          |0          |1          |1          |0|

Proportion of times we agree on relevant/not relevent: 0.82 out of 51 we both labelled.

|             |Liz tag 2  |Liz tag 5  |Liz tag 1  |Liz tag 3|
|---|---|---|---|---|
|Nonie tag 2          |0          |0          |0          |0|
|Nonie tag 5          |0         |30          |0          |0|
|Nonie tag 1          |1          |0          |9          |0|
|Nonie tag 3          |1          |0          |1          |0|


### Evaluation for Nonie's second go at grants tags

Proportion of times there was agreement: 0.93 out of 533 relabeled grants.

|                          |Nonie's second tag 1  |Nonie's second tag 0|
|---|---|---|
|Nonie's original tag 1                      |41                      |24|
|Nonie's original tag 0                      |13                     |455|

### Evaluation between Liz and Nonie's grants tags

Proportion of times there was agreement: 1.0 out of 27 relabeled grants.

|               |Liz tag 1.0  |Liz tag 5.0|
|---|---|---|
|Nonie tag 1.0            |4            |0|
|Nonie tag 5.0            |0           |23|

### Evaluation between Liz/Nonie and Becky's grants tags

Proportion of times there was agreement: 0.70 out of 76 relabeled grants.

|                   |Becky tag 1.0  |Becky tag 5.0|
|---|---|---|
|Nonie/Liz tag 1.0             |28              |2|
|Nonie/Liz tag 5.0             |21             |25|

## `grant_tagger.py`

In `grant_tagger.py` we train a model to predict whether a grant is relevant or not. For this we collapsed the classification into binary, where
- 1 = class was relevant tool (1), relevant model (2) or relevant dataset (3)
- 0 = class was not relevant (5)

## Model Results

These results are just from running one training of the model using a specific train/test split specified by default random number seeds. Thus there will be variation in the results if different seeds are used.

| Date | ngram range | Test proportion | Vectorizer type | Model type | Bert type (if relevant) | Not relevent sample size | Train size | Test size | Train F1 | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200806 | (1,2) | 0.25 | count | log_reg | - | 1.0 | 438 | 146 | 0.998 | 0.778 |
| 200806 | (1,2) | 0.25 | count | naive_bayes | - | 1.0 | 438 | 146 | 0.998 | 0.829 |
| 200806 | (1,2) | 0.25 | count | SVM | - | 1.0 | 438 | 146 | 0.982 | 0.803 |
| 200806 | (1,2) | 0.25 | tfidf | log_reg | - | 1.0 | 438 | 146 | 0.998 | 0.828 |
| 200806 | (1,2) | 0.25 | tfidf | naive_bayes | - | 1.0 | 438 | 146 | 0.991 | 0.718 |
| 200806 | (1,2) | 0.25 | tfidf | SVM | - | 1.0 | 438 | 146 | 0.998 | 0.759 |
| 200806 | (1,2) | 0.25 | bert | naive_bayes | bert | 1.0 | 438 | 146 | 0.754 | 0.667 |
| 200806 | (1,2) | 0.25 | bert | SVM | bert | 1.0 | 438 | 146 | 0.850 | 0.794 |
| 200806 | (1,2) | 0.25 | bert | log_reg | bert | 1.0 | 438 | 146 | 0.993 | 0.819 |
| 200806 | (1,2) | 0.25 | bert | naive_bayes | scibert | 1.0 | 438 | 146 | 0.809 | 0.735 |
| 200806 | (1,2) | 0.25 | bert | SVM | scibert | 1.0 | 438 | 146 | 0.821 | 0.738 |
| 200806 | (1,2) | 0.25 | bert | log_reg | scibert | 1.0 | 438 | 146 | 0.998 | 0.814 |
| 200807 | (1,2) | 0.25 | count | log_reg | - | 1.0 | 321 | 107 | 1.0 | 0.804 |
| 200807 | (1,2) | 0.25 | count | naive_bayes | - | 1.0 | 321 | 107 | 1.0 | 0.867 |
| 200807 | (1,2) | 0.25 | count | SVM | - | 1.0 | 321 | 107 | 0.975 | 0.745 |
| 200807 | (1,2) | 0.25 | tfidf | log_reg | - | 1.0 | 321 | 107 | 1.0 | 0.811 |
| 200807 | (1,2) | 0.25 | tfidf | naive_bayes | - | 1.0 | 321 | 107 | 1.0 | 0.862 |
| 200807 | (1,2) | 0.25 | tfidf | SVM | - | 1.0 | 321 | 107 | 1.0 | 0.729 |
| 200807 | (1,2) | 0.25 | bert | naive_bayes | bert | 1.0 | 321 | 107 | 0.679 | 0.758 |
| 200807 | (1,2) | 0.25 | bert | SVM | bert | 1.0 | 321 | 107 | 0.793 | 0.817 |
| 200807 | (1,2) | 0.25 | bert | log_reg | bert | 1.0 | 321 | 107 | 1.0 | 0.814 |
| 200807 | (1,2) | 0.25 | bert | naive_bayes | scibert | 1.0 | 321 | 107 | 0.740 | 0.835 |
| 200807 | (1,2) | 0.25 | bert | SVM | scibert | 1.0 | 321 | 107 | 0.738 | 0.796 |
| 200807 | (1,2) | 0.25 | bert | log_reg | scibert | 1.0 | 321 | 107 | 1.0 | 0.867 |


### Changes made

200807:
- Add a few new EPMC tags
- Add title and grant type to X data.
- Take out grants with no descriptions, and remove duplicates where the 6 digit grant number is the same and the description is the same

I expect the changes to the training data made most of the difference, but adding the title and grant types did contribute a little to the increases seen (I tested without these and the scores were slightly smaller).

## Ensemble model
We took the top performing models, labeled the grant data with each model and defined a grant as containing  a tool only if all three models agreed.

Criteria for chosing models:
F1 >= 0.8
Precision >= 0.82
Recall >= 0.82

This returned 3 models: bert_log_reg_scibert_200807, bert_naive_bayes_scibert_200807, count_naive_bayes_200807

It found 2163 grants

test results:

| Model | f1 | precision_score | recall_score |
| --- | --- | --- | --- |
| bert_log_reg_scibert_200807 |	0.866666667 | 0.825396825 | 0.912280702 |
| bert_naive_bayes_scibert_200807 | 0.834782609	|0.827586207 | 0.842105263 |
| count_naive_bayes_200807 | 0.866666667 | 0.825396825 | 0.912280702 |
| Ensemble (mean) | 0.856038647 | 0.826126619 | 0.888888889 |


# 22nd October 2020

There are 2 random number seeds in the creation of the training data - one for selecting the values from the irrelevant dataset to use, and one for splitting of the training and test datasets. This process has been rewritten so that it only uses one random seed.

## The best random seed and variability in results

There is quite a bit of variability in the model results due to which random seed you use to split the data into the training and test sets. To scope how large this variability is and which random seed might generally produce good results on the different models we ran each model type 10 times with different random seeds.

By running

```
python nutrition_labels/grant_tagger_seed_experiments.py
```
different combinations of the vectorizer and model types will be run 10 times and the results outputted.

We used the training data `data/processed/training_data/200807/training_data.csv` for all these experiments.


### Variability in the results:

| Model | Mean/std/range test accuracy | Mean/std/range test f1 | Mean/std/range test precision_score | Mean/std/range test recall_score |
| ----- | ------------------ | ------------ | ------------------------- | ---------------------- |
| count_naive_bayes_201021|0.793/ 0.044/ (0.701, 0.85)|0.805/ 0.040/ (0.724, 0.864)|0.780/ 0.063/ (0.609, 0.833)|0.839/ 0.070/ (0.732, 0.927)|
| count_log_reg_201022|0.778/ 0.035/ (0.729, 0.832)|0.783/ 0.033/ (0.729, 0.842)|0.778/ 0.036/ (0.696, 0.824)|0.791/ 0.055/ (0.696, 0.873)|
| tfidf_naive_bayes_201021|0.750/ 0.078/ (0.57, 0.822)|0.777/ 0.059/ (0.662, 0.846)|0.730/ 0.105/ (0.506, 0.821)|0.855/ 0.100/ (0.644, 0.957)|
| tfidf_log_reg_201021|0.776/ 0.054/ (0.701, 0.841)|0.767/ 0.062/ (0.681, 0.844)|0.817/ 0.067/ (0.651, 0.9)|0.737/ 0.121/ (0.571, 0.885)|
| bert_naive_bayes_bert_201021|0.731/ 0.053/ (0.636, 0.804)|0.737/ 0.051/ (0.636, 0.817)|0.736/ 0.071/ (0.61, 0.812)|0.744/ 0.065/ (0.596, 0.825)|
|bert_SVM_bert_201022|0.767/ 0.042/ (0.72, 0.869)|0.785/ 0.036/ (0.754, 0.881)|0.743/ 0.052/ (0.647, 0.825)|0.838/ 0.063/ (0.763, 0.945)|
|bert_log_reg_bert_201022|0.759/ 0.037/ (0.71, 0.813)|0.761/ 0.037/ (0.713, 0.825)|0.771/ 0.066/ (0.65, 0.894)|0.761/ 0.074/ (0.643, 0.855)|
|bert_log_reg_scibert_201022|0.783/ 0.035/ (0.738, 0.832)|0.791/ 0.030/ (0.727, 0.826)|0.782/ 0.045/ (0.738, 0.865)|0.806/ 0.073/ (0.643, 0.894)|
|count_SVM_201022|0.753/ 0.043/ (0.701, 0.841)|0.750/ 0.048/ (0.687, 0.847)|0.772/ 0.045/ (0.691, 0.839)|0.736/ 0.084/ (0.607, 0.855)|
|tfidf_SVM_201022|0.751/ 0.062/ (0.654, 0.832)|0.721/	0.089/ (0.584, 0.833)|0.843/ 0.081/ (0.656, 0.946)|0.660/ 0.178/ (0.441, 0.894)|
|bert_SVM_scibert_201022|0.776/ 0.039/ (0.748, 0.879)|0.780/ 0.038/ (0.75, 0.879)|0.787/ 0.080/ (0.656, 0.904)|0.784/ 0.080/ (0.684, 0.904)|


### Best seed:
We calculated the highest average metrics over all models for the different random seeds used.

The highest average f1 score over all models (0.838) was found when the random seed 0 was used, followed by 5 (0.797) and then 4 (0.784). The highest average precision score over all models (0.836) was found when the random seed 2 was used, followed by 0 (0.820) and then 1 (0.806). The highest average recall score over all models (0.870) was found when the random seed 9 was used, followed by 5 (0.864) and then 0 (0.861).

Thus we will take the **'best seed' to be 0** since it consistently produces high scores regardless of metric.

### Rerunning all the Models

ngram range: (1,2),
Test proportion : 0.25,
Train size: 321,
Test size: 107,
Not relevant ratio: 1

| Date | Vectorizer type | Model type | Bert type (if relevant) | Train F1 | Test F1 | Test precision | Test recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 201022 | count | log_reg | - | 1.000 | 0.842 | 0.814 | 0.873 |
| 201022 | count | naive_bayes | - | 1.000 | 0.864 | 0.810 | 0.927 |
| 201022 | count | SVM | - | 0.994 | 0.847 | 0.839 | 0.855 |
| 201022 | tfidf | log_reg | - | 1.000 | 0.844 | 0.852 | 0.836 |
| 201022 | tfidf | naive_bayes | - | 1.000 | 0.846 | 0.765 | 0.945 |
| 201022 | tfidf | SVM | - | 1.000 | 0.822 | 0.846 | 0.800 |
| 201022 | bert | naive_bayes | bert | 0.713 | 0.757 | 0.813 | 0.709 |
| 201022 | bert | SVM | bert | 0.819 | 0.881 | 0.825 | 0.945 |
| 201022 | bert | log_reg | bert | 1.000 | 0.825 | 0.797 | 0.855 |
| 201022 | bert | naive_bayes | scibert | 0.772 | 0.796 | 0.811 | 0.782 |
| 201022 | bert | SVM | scibert | 0.776 | 0.879 | 0.904 | 0.855 |
| 201022 | bert | log_reg | scibert | 1.000 | 0.814 | 0.762 | 0.873 |

### Ensemble model

Running:
```
python nutrition_labels.ensemble_model.py
```
gives us the results of using an ensemble of high performing models. The script outputs the results for tech grant classification when different numbers of models need to agree.

Only choosing models which were:
- Trained on 201022
- F1 >= 0.8
- Precision >= 0.82
- Recall >= 0.82

This returned 4 models: ['count_SVM_201022', 'bert_SVM_scibert_201022', 'bert_SVM_bert_201022', 'tfidf_log_reg_201022']

We can experiment with different numbers of the models needing to agree and find how well the model does:

| Number of models that need to agree | Number of relevant grants | Test F1 | Test precision | Test recall |
|---|---|---|---|---|
|1|5926|0.860|0.788|0.945|
|2|4125|0.885|0.862|0.909|
|3|2956|0.873|0.873|0.873|
|4|1257|0.832|0.913|0.764|

All the results for this are in the `data/processed/ensemble/` folder with the '201118' tag.

## Tech grants - final results

After some fairness analysis in [the fairness notebook](Tech_grant_model_fairness.md), we decided to use an ensemble of models to make predictions on whether a grant was likely to produce ‘tech’. The model chosen was one where 3 out of 4 models (bert_SVM, scibert_SVM, count_SVM, and TFIDF_logreg) need to agree on a grant being a tech grant in order for the final prediction to be classified as a tech grant.

Using this model there were 2956 grants predicted to be tech grants. Summary details from a large subsection of these grants (where grants from the same family are collapsed into one grant) is shown in our Wellcome’s grants Tableau dashboard ([here](https://tableau.wellcome.org/#/views/Wellcome/WellcomeOverview/GallaghE@wellcomeit.com/TechGrants-ensemble3_201118?:iid=1), but may not be available to everyone). A still from this is shown below:

![](figures/ensemble_grants_201118.png)

### Comparison with the Science tagging for tech tags

Wellcome also has a model for tagging grants (we'll call this "the Science tags") with, so we wanted to compare our tech grant tags with theirs. We normalised both sets of data so that the same range of years were used (2005-2019). The Science tags include many different topics, and we selected the 'techy' topic tags: Data Science, Computational & Mathematical Modelling and/or Surveillance (we’ll call these the computational science tag grants).​ The Science tags come with a probability, and we were advised to only include the tags where the probability is over 0.4. This comparison was done in the notebook `Science tags - Tech grant comparison.ipynb`.

In general there wasn't a huge overlap between the two sets of tags, as seen in the venn plot:

![](figures/tag_comparisons_venn.png)

A deeper dive showed that the proportions of grant types in both sets of tags would have been broadly similar had it not been skewed by a large number of PhD studentships in the tech grants but not in the computational science tags. Furthermore, looking at the proportion of grants in both sets of tags by grant year showed that there are more recent grants in the computational science tags.

![](figures/tag_comparisons_granttype.png)
![](figures/tag_comparisons_years.png)
