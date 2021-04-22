#!/bin/bash
export PYTHONPATH='.'
echo 'Creating training data ...'
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.08.test.ini
echo 'Creating evaluation data ...'
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.test.epmc.ini
python nutrition_labels/create_training_data.py --config_path configs/training_data/2021.03.29.test.rf.ini
echo 'Training models ...'
python nutrition_labels/grant_tagger.py --config_path configs/train_model/2021.04.02.test.ini
echo 'Predicting tech grants ...'
python nutrition_labels/predict.py --config_path configs/predict/2021.04.02.test.ini
echo 'Evaluating tech grants ...'
python nutrition_labels/evaluate.py --config_path configs/evaluation/2021.04.02.test.ini