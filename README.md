# Tech Wellcome Funds and Data Representation Labels

This repo contains the exploratory notebooks, modelling and analysis scripts to find out what tech Wellcome funds, and also the code to create the data representation labels.

## Set-up
### Set up the virtual environment

Running
```bash
make virtualenv
```
will create a virtual environment with the packages listed in `requirements.txt`.

Then when you want to develop and run code start this up by running
```bash
source build/virtualenv/bin/activate
```

### Download the data

Download the data for this project from
```
https://wellcomecloud.sharepoint.com/:f:/r/sites/DataLabs/Machine%20Learning/Nutrition%20Labels/data?csf=1&web=1&e=XHFn6n
```
you will need to have been granted access to access this folder.

Make sure to upload any processed data to this folder too.

### Jupyter notebooks

To run a notebook run
```bash
jupyter notebook
```

## Tech Wellcome Funds

The code for this project is in the `nutrition_labels` and `notebooks` folder. More information about the experiments and results to this project are given in the documents:
- [Finding_Tech_Grants.md](docs/Finding_Tech_Grants.md)
- [Tech_grant_model_fairness.md](docs/Tech_grant_model_fairness.md)
- [Tech_grant_clusters.md](docs/Tech_grant_clusters.md)

### Creating training data

In [Finding_Tech_Grants.md](docs/Finding_Tech_Grants.md) we describe the first stage of tagging data to create the training data - this was last updated on the 7th August 2020. In 2021 we updated the definition of 'tech' as described in [Expanding_tech_grants.md](docs/Expanding_tech_grants.md) - this created a new training data set on 26th January 2021. Then, we decided to use active learning in Prodigy to tag more training data, this process is described in [Prodigy_training_data.md](docs/Prodigy_training_data.md), this resulted in a training data set on the 21st February 2021.

Finally, we found that adding grants tagged via EPMC and ResearchFish actually may decrease the model scores, so we created some training data not containing these data points - this was done on 8th March 2021.

| Tag code | Meaning | Number of grants - 200807 | Number of grants - 210126 | Number of grants - 210221 | Number of grants - 210308|
|---|---|--- |--- | --- | --- |
| 1 | Relevant | 214 |347 | 495 | 313 |
| 0 | Not relevant | 883 |349 | 485 | 488 |

To create these datasets you should run:
```
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.08.ini
```
with config files from '2020.08.07.ini', '2021.01.26.ini', '2021.02.21.ini' or 2021.03.08.ini'.


### Training a tech tagging model

You can train and save a model, or several models, to classify grants as being to do with tech or not (see definitions for this in [Finding_Tech_Grants.md](docs/Finding_Tech_Grants.md)) by creating a config file which contains various training arguments for the experiments. The main arguments in this config file are:

- `training_data_file`, e.g. `data/processed/training_data/210308/training_data.csv`, the training data file.
- `vectorizer_types`, e.g. `['count', 'tfidf', 'bert', 'scibert']`, the vectorizers you want to use.
- `classifier_types`, e.g. `['naive_bayes', 'SVM', 'log_reg']`, the classifier types you want to use.

Then by running 
```
python nutrition_labels/grant_tagger.py --config_path configs/train_model/2021.03.16.ini
```
every combination from `vectorizer_types` and `classifier_types` will be run.

This will create a folder in `models/` named after the config version, and each trained model will be stored in their own subfolders. A summary json file with the model predictions will be stored in e.g. `models/210316/training_information.json`, this will be important in evaluating the ensemble model.

### Fairness

In the notebook [Fairness.ipynb](notebooks/Fairness.ipynb) we perform group fairness checks for the models. The results of these are written up in [Tech_grant_model_fairness.md](docs/Tech_grant_model_fairness.md).

### Computational Science tags comparison

We compare the tech grants with another set of grants tagged by a different model in the notebook [Science tags - Tech grant comparison.ipynb](notebooks/Science%20tags%20-%20Tech%20grant%20comparison.ipynb").

### Clustering

We perform cluster analysis to look at themes within the tech grants. This analysis is written up in [Tech_grant_clusters.md](docs/Tech_grant_clusters.md).

## Using the tech grants tagger

Create a config file with the name of the grants data csv file you want to predict on, the names of the columns of text, and which models to use and how many need to agree (see `configs/ensemble/2020.02.21.ini` for structure). Then input this config as an argument to `ensemble_grant_tagger.py`, for example:

```
python nutrition_labels/ensemble_grant_tagger.py --config_path configs/ensemble/2020.02.21.ini
```

This will output a csv file of grant ID - tech or not predictions.


## Data Representation Labels

In the [Top_Datasets.md](docs/Top_Datasets.md) document you can find how we identified some of the datasets to include in the data representation labels.

The folder `representation_labels/` contains the code needed to produce the data representation labels html.

## Project structure

```
├── configs            
│   Config files for the models
├── data
│   ├── processed - Data that we generate     
│   └── raw - Raw data                    
├── docs            
│   Documents for this project
├── models            
│   Any models created including clustering
├── notebooks                
|   Jupyter notebooks for experimentation and analysis
├── nutrition_labels
│   Scripts for the tech wellcome funds work
├── README.md
├── representation_labels
│   Scripts for the data representation labels work
├── requirements.txt - a list of all the python packages needed for this project  

```
