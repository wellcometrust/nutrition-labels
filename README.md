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

### Training a tech tagging model

You can train and save a model to classify grants as being to do with tech or not (see definitions for this in [Finding_Tech_Grants.md](docs/Finding_Tech_Grants.md)) by running:

```
python nutrition_labels/grant_tagger.py --training_data_file data/processed/training_data.csv --vectorizer_type count --relevant_sample_ratio 1 --model_type naive_bayes --bert_type bert
```

where `vectorizer_type` can be 'count', 'tfidf' or 'bert', `model_type` can be 'naive_bayes', 'SVM' and 'log_reg', and `bert_type` (if using bert) can be 'bert' and 'scibert'.

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
