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

Sync the data from S3 by running:
```
make sync_data_from_s3
```

And if you want to upload new data to the S3 location for this project run:
```
make sync_data_to_s3
```
which will sync every file in the `data/` folder.

## Jupyter notebooks

To run a notebook run
```bash
source build/virtualenv/bin/activate
jupyter notebook
```

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

## Raw Data Codes 

Each entry in the raw data sets have been given a code to identify them as useful entries 
They are:
EPMC_relevant_tools_pubs_manual_edit and research_fish_manual_edit
* 1 = medical data set tool
* 2 = medical data set model
* 3 = medical data set tool
* 4 = maybe relavent need more info
* 5 = not relevent 
* 6 = edge case 

wellcome-grants-awarded-2005-2019_manual_edit
* 1 = Relevent medical data set, tool or model
* 2 = Possibly relevent medical data set, tool or model
* 3 = Not relevent medical data set, tool or model
* 4 = Not relevent medical data set, tool or model


