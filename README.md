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

Download the data for this project from 
```
https://wellcomecloud.sharepoint.com/:f:/r/sites/DataLabs/Machine%20Learning/Nutrition%20Labels/data?csf=1&web=1&e=XHFn6n
```
you will need to have been granted access to access this folder.

Make sure to upload any processed data to this folder too.

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