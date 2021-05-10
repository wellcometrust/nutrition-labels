#!/bin/bash
export PYTHONPATH='.'

set -e

# get config from input
CONFIG=$1

echo "Running pipeline for ${CONFIG}"

echo 'Creating training data ...'
python nutrition_labels/create_training_data.py --config_path ${CONFIG}

echo 'Training models ...'
python nutrition_labels/grant_tagger.py --config_path ${CONFIG}

echo 'Predicting tech grants ...'
python nutrition_labels/predict.py --config_path ${CONFIG}

echo 'Evaluating tech grants ...'
python nutrition_labels/evaluate.py --config_path ${CONFIG}