# Tech Wellcome Funds and Data Representation Labels

This repo contains the exploratory notebooks, modelling and analysis scripts to find out what tech Wellcome funds, and also the code to create the data representation labels.

### Contents
1. [Set-up](#setup)
2. [Tech Wellcome Funds](#tech)
  - [Pipeline overview](#pipelineoverview)
    - [Create the training data](#pipeline1)
    - [Train models](#pipeline2)
    - [Predict tech grants using a single or an ensemble of models](#pipeline3)
    - [(optional) Create evaluation data](#pipeline4)
    - [(optional) Evaluate the model](#pipeline5)
  - [Pipeline additional details](#pipelineadditional)
    - [Creating training data](#trainingdata)
    - [Model Experiments](#experiments)
    - [Fairness](#fairness)
    - [Computational Science tags comparison](#comparison)
    - [Clustering](#clustering)
3. [Data Representation Labels](#labels)
4. [Project structure](#structure)

## Set-up <a name="setup"></a>

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

If you have the AWS command line tool and added your configure credentials then you can either download the open version of the data and models folder from our publically available S3 bucket:
```
make sync_open_data_from_s3
make sync_open_models_from_s3
```
(this only contains the essential and most recent files), or (if a Wellcome employee) download the private version of all the data and models:
```
make sync_data_from_s3
make sync_models_from_s3
```
or for a smaller set (no legacy data/models) of private data: 
```
make sync_latest_files_from_s3
```

If you don't have AWS credentials then you will need to download the essential open files by going to this url:
```
https://datalabs-public.s3.eu-west-2.amazonaws.com/nutrition-labels/open_data_models.zip
```
from this a zipped file will be downloaded. You will need to unzip this file and move the contents of this folder into the main directory (i.e. the 'data/processed' folder this zipped file should replace the main 'data/processed' folder).


This data contains the file `data/raw/wellcome-grants-awarded-2005-2019.csv` which is the openly available 360Giving grants data of 16,914 grants from 2005 to 2019. This file is the basis of a lot of this project.

Make sure to upload any processed data to this folder too.

### Jupyter notebooks

To run a notebook run
```bash
jupyter notebook
```

### Tests

Unit tests can be run by running `make test`. After any changes made to this codebase this command should be ran in order to check nothing has been broken.

To check the pipeline of training models and predicting grants is working can be done with:
```
chmod +x pipelines/tech_grants_pipeline.sh
pipelines/tech_grants_pipeline.sh configs/pipeline/2021.05.01.test.ini
```
this should take about 1 min.

## Tech Wellcome Funds <a name="tech"></a>

The code for this project is in the `nutrition_labels` and `notebooks` folder. More information about the experiments and results to this project are given in the documents:
- [Finding_Tech_Grants.md](docs/Finding_Tech_Grants.md)
- [Tech_grant_model_fairness.md](docs/Tech_grant_model_fairness.md)
- [Tech_grant_clusters.md](docs/Tech_grant_clusters.md)
- [Expanding_tech_grants.md](docs/Expanding_tech_grants.md)
- [Prodigy_training_data.md](docs/Prodigy_training_data.md)
- [Tech_Grants_2021.md](docs/Tech_Grants_2021.md) - the most current experiment descriptions are all in this document.

### Pipeline overview <a name="pipelineoverview"></a>

#### ResearchFish and EPMC evaluation data

Create RF and EPMC evaluation data by running:

`python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.epmc.ini`
and 
`python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.rf.ini`

Previously these commands were part of the pipeline bash command, but they require different config parameters to `create_training_data.py` - which isn't possible now since we are using one config for the entire pipeline (so there are different config parameters).

#### Pipeline bash command

The pipeline to create training data, train models, and make predictions and evaluate models can be run with the commands:
```
chmod +x pipelines/tech_grants_pipeline.sh
pipelines/tech_grants_pipeline.sh configs/pipeline/2021.05.01.private.ini
```
or
```
chmod +x pipelines/tech_grants_pipeline.sh
pipelines/tech_grants_pipeline.sh configs/pipeline/2021.05.01.open.ini
```
The former uses internally available FortyTwo data to train and make predictions on, the second command is for external users - it trains and predicts on the publically available 360Giving grants dataset.

Be warned that this takes >5 hours since it includes making predictions on data.

An overview of these pipeline steps (if you were to run them one by one) and the latest files, as of 21/04/2021 used for each of them is as follows.

##### 1. Create the training data <a name="pipeline1"></a>
Description: Create training data with expanded tech definition and tagged grants data. ResearchFish and EPMC data are not included in the data.

Input: `configs/training_data/2021.03.08.ini`

Command:
```
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.08.ini
```
Output: 313 tech grants and 488 not tech grants (Internal ID <-> Relevance code) `data/processed/training_data/210308/training_data.csv`.

##### 2. Train model(s) <a name="pipeline2"></a>
Description: Train a BERT + logistic regression classifier model.

Input: Internally we want to train and predict on fortytwo data, but for external communication we need to use the public 360 giving dataset. So we have different configs to train on each of these:
- `configs/train_model/2021.04.03.ini` - to train using the fortytwo grants data downloaded on 20th April 2021.
- `configs/train_model/2021.04.04.ini` - to train using the 360 giving grants dataset.

Command example:
```
python nutrition_labels/grant_tagger.py --config_path configs/train_model/2021.04.03.ini
```
Output:
All outputs stored in `models/210403/`, the pickled trained classifier and vectorizer is stored in the folder `models/210403/bert_log_reg_210403/` along with a `evaluation_results.txt` file with the test/train metrics. Another file is stored in the main directory `models/210403/training_information.json` which contains information of which data points were in the test/train split and what the model predicted.

##### 3. Predict tech grants using a single or an ensemble of models <a name="pipeline3"></a>
Description: Predict whether grants should be classified as tech grants or not, using the trained `models/210403/bert_log_reg_210403` model with a prediction threshold of 0.55.

Input: Internally we want to predict on fortytwo data, but for external communication we need to use the public 360 giving dataset. So we have different configs to predict on each of these:
- `configs/predict/2021.04.04.ini` - to predict using the 360 giving grants dataset.
- `configs/predict/2021.04.05.ini` - to predict using the fortytwo grants data downloaded on 20th April 2021.

Command:
```
python nutrition_labels/predict.py --config_path configs/predict/2021.04.04.ini
```
```
python nutrition_labels/predict.py --config_path configs/predict/2021.04.05.ini
```
Output:
- `data/processed/predictions/210404/wellcome-grants-awarded-2005-2019_tagged.csv`
- `data/processed/predictions/210405/all_grants_fortytwo_info_210420_tagged.csv`

##### 4. (optional) Create evaluation data <a name="pipeline4"></a>
Description: A sample of ResearchFish (self-reported) and EPMC (publications) outputs data was tagged as containing a tech output. This can be seen as 'hidden' tech - and so the grants description may not have any mention of tech. It is interesting to see how well the model does on predicting these grants, but first this needs to be processed.

Input:
- `configs/training_data/2021.03.29.epmc.ini`
- `configs/training_data/2021.03.29.rf.ini`

Command:
```
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.epmc.ini
```
and

```
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.rf.ini
```

Output:
- `data/processed/training_data/210329epmc/training_data.csv`
- `data/processed/training_data/210329rf/training_data.csv`

##### 5. (optional) Evaluate the model <a name="pipeline5"></a>
Description: The model is evaluated on up to 4 different datasets - the test data, unseen grants data containing only not-tech grants, and the tech outputs from the EPMC and ResearchFish outputs data. The unseen not-tech grants data was the disregarded set of training data to make the test and training datasets have a balanced number of tech and not tech grants. If step 4 wasn't done, this will still work and just output the test and unseen data metrics.

Input: `configs/evaluation/2021.04.03.ini`

Command:
```
python nutrition_labels/evaluate.py --config_path configs/evaluation/2021.04.03.ini
```
Output:
Metrics for all 4 evaluation data sets are outputted in `data/processed/evaluation/210403/evaluation_results.txt`.

### Pipeline additional details <a name="pipelineadditional"></a>
#### Creating training data <a name="trainingdata"></a>

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

#### Model Experiments <a name="experiments"></a>

Several experiments were performed to come up with the best model for this task. The outcomes of these fed into the design of each of the config files in the pipeline. These experiments are discussed in much more detail in the "Experiments" and "Ensemble model/Parameter experiments" sections of [Tech_Grants_2021.md](docs/Tech_Grants_2021.md).

#### Fairness <a name="fairness"></a>

In the notebook [Fairness.ipynb](notebooks/Fairness.ipynb) we perform group fairness checks for the models. The results of these are written up in [Tech_grant_model_fairness.md](docs/Tech_grant_model_fairness.md).

#### Computational Science tags comparison <a name="comparison"></a>

We compare the tech grants with another set of grants tagged by a different model in the notebook [Science tags - Tech grant comparison.ipynb](notebooks/Science%20tags%20-%20Tech%20grant%20comparison.ipynb").

#### Clustering <a name="clustering"></a>

We perform cluster analysis to look at themes within the tech grants. This analysis is written up in [Tech_grant_clusters.md](docs/Tech_grant_clusters.md).

# Data Representation Labels <a name="labels"></a>

In the [Top_Datasets.md](docs/Top_Datasets.md) document you can find how we identified some of the datasets to include in the data representation labels.

The folder `representation_labels/` contains the code needed to produce the data representation labels html.

# Project structure <a name="structure"></a>

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

## Deploying

### Pre-requisites

You need docker installed and to download the model-cli from https://github.com/wellcometrust/hal9000/releases/tag/cli-0.1.0.

To deploy a model, change the code, run `make run-debug`.
