#!/bin/bash
export PYTHONPATH='.'
echo 'Creating training data ...'
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.08.ini
echo 'Creating evaluation data ...'
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.epmc.ini
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.rf.ini
echo 'Training models ...'
python nutrition_labels/grant_tagger.py --config_path configs/train_model/2021.04.03.ini
echo 'Predicting tech grants ...'
python nutrition_labels/predict.py --config_path configs/predict/2021.04.04.ini
echo 'Evaluating tech grants ...'
python nutrition_labels/evaluate.py --config_path configs/evaluation/2021.04.03.ini