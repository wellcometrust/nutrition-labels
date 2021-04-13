"""
Usage
----------
python nutrition_labels/grant_tagger.py 
    --config_path configs/train_model/2021.03.16.ini

Description
----------
This code will train several models (as specified in the config argument) using the same 
training data and parameters inputted in the config file. 
The trained models and their evaluation results, as well as overall training information 
will be saved in folder unique to the config file e.g 'models/210316/'

Config notes
----------
split_seed in the config file can be kept blank "split_seed = "
In this case the seed is different for every model trained, which means
the training data will be different for each model and so to evaluate
an ensemble model properly you will need a hold-out test set which hasn't
been used in the training of any of the models.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from wellcomeml.ml.bert_vectorizer import BertVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np

from random import sample, seed, shuffle
from argparse import ArgumentParser
from datetime import datetime
import pickle
import configparser
import os
import ast
import json

from nutrition_labels.utils import pretty_confusion_matrix, clean_string


class GrantTagger:
    """
    A class to train a binary classifier for grants text.
    ...

    Attributes
    ----------
    test_size : float (default 0.25)
        Proportion of data to use in test set.
    relevant_sample_ratio : float (default 1)
        The ratio of tech (1) data points to not-tech (0) data points
        to include in the test/train data. The remainder is disgarded.
        This is not random - if not-tech points need to be disgarded they
        are selected uniformly through the data.
    split_seed : int (default 1)
        The seed used to randomly shuffle the data for the test/train split.
    vectorizer_type : str (default "count")
        The vectorizer to use, can be from ['count', 'tfidf', 'bert', 'scibert'] 
    classifier_type : str (default "naive_bayes")
        The classifier to use, can be from ['naive_bayes', 'SVM', 'log_reg'] 
    prediction_cols : list (default ["Title", "Grant Programme:Title", "Description"])
        The column names for the text you want to train the model on. If len > 1 then
        they will be merged into one. 
    label_name : str (default "Relevance code")
        The column name of the classification truth label.
    pred_prob_threshold : float (default None)
        A prediction probability threshold that needs to be satisfied for a datapoint
        to be predicted as tech.

    Methods
    -------
    fit_transform(data, train_data_id)
        Fit a vectorizer using the text data
    transform(data)
        Vectorize text data using the fitted vectorizer
    split_data(X_vect, y)
        Split the data into train and test sets
    fit(X, y)
        Train the model
    predict(X)
        Make predictions using the trained model
    predict_proba(X)
        Output prediction probabilities using the trained model
    evaluate(X, y, extra_scores, average)
        Evaluate various metrics using the model.
    train_test_info()
        Create an output dictionary with information about
        which data points were used in the test or training, and
        the model predictions for each.
    save_model(output_path, evaluation_results)
    load_model(output_path)
    """

    def __init__(
        self,
        test_size=0.25,
        relevant_sample_ratio=1,
        split_seed=1,
        vectorizer_type="count",
        classifier_type="naive_bayes",
        prediction_cols=["Title", "Grant Programme:Title", "Description"],
        label_name="Relevance code",
        pred_prob_threshold=None,
    ):
        self.test_size = test_size
        self.relevant_sample_ratio = relevant_sample_ratio
        self.split_seed = split_seed
        self.vectorizer_type = vectorizer_type
        self.classifier_type = classifier_type
        self.prediction_cols = (*prediction_cols,)
        self.label_name = label_name
        self.pred_prob_threshold = pred_prob_threshold

    def process_grant_text(self, data):
        """
        Create a new column of a pandas dataframe which included
        the merged and cleaned text from multiple columns.
        """
        data.fillna('', inplace=True)
        data["Grant texts"] = data[list(self.prediction_cols)].agg(
            ". ".join, axis=1
        ).apply(clean_string)

        return data

    def transform(self, data):
        """
        Vectorize the joined text from columns of a pandas dataframe ('data'),
        using a fitted vectorizer.
        """
        if "Grant texts" not in data:
            data = self.process_grant_text(data)

        X = data["Grant texts"].tolist()
        X_vect = self.vectorizer.transform(X)

        if self.scaler:
            X_vect = self.scaler.transform(X_vect)
            
        return X_vect

    def fit_transform(self, data, train_data_id="Internal ID"):

        if "Grant texts" not in data:
            # If the training data hasn't come through Prodigy tagging then this won't exist
            data = self.process_grant_text(data)

        self.X_ids = data[train_data_id].tolist()
        self.X = data["Grant texts"].tolist()
        self.y = data[self.label_name].tolist()

        if self.vectorizer_type == "count":
            self.vectorizer = CountVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                stop_words='english'
            )
        elif self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                stop_words='english'
            )
        elif "bert" in self.vectorizer_type:
            self.vectorizer = BertVectorizer(pretrained=self.vectorizer_type)
        else:
            print("Vectorizer type not recognised")
        self.X_vect = self.vectorizer.fit_transform(self.X)

        # For BERT/SciBERT + naive bayes the values need to be between 0 and 1, 
        # so scale the values.
        self.scaler = None
        if "bert" in self.vectorizer_type and self.classifier_type == "naive_bayes":
            self.scaler = MinMaxScaler()
            self.X_vect = self.scaler.fit_transform(self.X_vect)

        return self.X_vect, self.y

    def split_data(self, X_vect, y):

        # Randomly shuffle the data
        random_index = list(range(len(y)))
        seed(self.split_seed)
        shuffle(random_index)
        if self.vectorizer_type == "bert":
            X_vect = [X_vect[i] for i in random_index]
        else:
            X_vect = X_vect[random_index]
        y = [y[i] for i in random_index]

        relevant_sample_index = [ind for ind, x in enumerate(y) if x != 0]
        irrelevant_sample_index = [ind for ind, x in enumerate(y) if x == 0]
        sample_size = int(
            round(len(relevant_sample_index) * self.relevant_sample_ratio)
        )
        if sample_size < len(irrelevant_sample_index):
            # Take sample_size equally spaced irrelevant points
            idx = np.round(
                np.linspace(0, len(irrelevant_sample_index) - 1, sample_size)
            ).astype(int)
            sample_index = relevant_sample_index + [
                irrelevant_sample_index[i] for i in idx
            ]
            # Make sure they are in order otherwise it'll be all 1111s and then 0000s
            sample_index.sort()
            y = [y[i] for i in sample_index]
            if self.vectorizer_type == "bert":
                X_vect = [X_vect[i] for i in sample_index]
            else:
                X_vect = X_vect[sample_index]

        # Randomly shuffled at the beginning so turn this off
        X_train, X_test, y_train, y_test = train_test_split(
            X_vect, y, test_size=self.test_size, shuffle=False
        )

        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        if self.classifier_type == "naive_bayes":
            model = MultinomialNB()
        elif self.classifier_type == "SVM":
            model = SVC(probability=True)
        elif self.classifier_type == "log_reg":
            model = LogisticRegression(max_iter=1000)
        else:
            print("Model type not recognised")
        self.model = model.fit(X, y)

    def predict(self, X):
        y_predict = self.model.predict(X).astype(int)
        if self.pred_prob_threshold:
            # If the prediction probability is over a threshold then allow a 1
            # prediction to stay as 1, otherwise switch to 0.
            # A prediction of 0 will stay at 0 regardless of probability.
            pred_probs = self.model.predict_proba(X)
            y_predict = y_predict*(np.max(pred_probs, axis=1) >= self.pred_prob_threshold)

        return y_predict

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y, extra_scores=False, average="binary"):

        y_predict = self.model.predict(X)

        scores = {
            "size": len(y),
            "accuracy": accuracy_score(y, y_predict),
            "f1": f1_score(y, y_predict, average=average),
            "precision_score": precision_score(
                y, y_predict, zero_division=0, average=average
            ),
            "recall_score": recall_score(
                y, y_predict, zero_division=0, average=average
            ),
        }

        if extra_scores:
            scores["Classification report"] = classification_report(y, y_predict)
            scores["Confusion matrix"] = pretty_confusion_matrix(y, y_predict)

        return scores

    def train_test_info(self):
        """
        Output a dict of information about the trained model's use of the data
        and the predictions and probabilities given, e.g
        {
            '123456/X/13/A': {
                'Truth': 1,
                'Prediction': 1,
                'Prediction probability': 0.7,
                'Test/train': 'Test'
            }
        }
        """

        y_predict = self.model.predict(self.X_vect)
        y_predict_proba = self.model.predict_proba(self.X_vect)
        # The probability of the predicted binary value
        y_predict_proba = np.max(y_predict_proba, axis=1)

        X_train_ids, X_test_ids, _, _ = self.split_data(np.array(self.X_ids), self.y)

        training_info = {}
        for grant_id, actual, pred, pred_prob in zip(
            self.X_ids, self.y, y_predict, y_predict_proba
        ):
            if grant_id in X_train_ids:
                split_type = "Train"
            elif grant_id in X_test_ids:
                split_type = "Test"
            else:
                split_type = "Not used"
            training_info[grant_id] = {
                "Truth": int(actual),
                "Prediction": int(pred),
                "Prediction probability": pred_prob,
                "Test/train": split_type,
            }

        return training_info

    def save_model(self, output_path, evaluation_results=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, "model.pickle"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(output_path, "vectorizer.pickle"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        if self.scaler:
            with open(os.path.join(output_path, "vectorizer_scaler.pickle"), "wb") as f:
                pickle.dump(self.scaler, f)

        if evaluation_results:
            evaluation_results["Vectorizer type"] = self.vectorizer_type
            evaluation_results["Classifier type"] = self.classifier_type
            evaluation_results["Split seed"] = self.split_seed
            with open(os.path.join(output_path, "evaluation_results.txt"), "w") as f:
                for key, value in evaluation_results.items():
                    f.write("\n" + key + ": " + str(value))

    def load_model(self, output_path):
        with open(os.path.join(output_path, "model.pickle"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(output_path, "vectorizer.pickle"), "rb") as f:
            self.vectorizer = pickle.load(f)
        try:
            with open(os.path.join(output_path, "vectorizer_scaler.pickle"), "rb") as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            self.scaler = None



def load_training_data(config, prediction_cols, train_data_id):

    training_data = pd.read_csv(config["data"]["training_data_file"])

    if "grants_text_data_file" in config["data"]:
        # Merge training data with another dataset containing the
        # text columns for predicting on
        grants_data = pd.read_csv(config["data"]["grants_text_data_file"])
        right_id = config["data"]["grants_text_data_file_id"]
        training_data = pd.merge(
            training_data,
            grants_data.drop_duplicates(subset=right_id)[
                prediction_cols + [right_id]
            ],
            how="left",
            left_on=train_data_id,
            right_on=right_id,
        )

    return training_data


def train_several_models(config):

    # Load data and parameters
    vectorizer_types = ast.literal_eval(config["models"]["vectorizer_types"])
    classifier_types = ast.literal_eval(config["models"]["classifier_types"])

    relevant_sample_ratio = config.getfloat("params", "relevant_sample_ratio")
    test_size = config.getfloat("params", "test_size")
    split_seed = config["params"].getint("split_seed")

    prediction_cols = ast.literal_eval(config["data"]["prediction_cols"])
    label_name = config["data"]["label_name"]
    train_data_id = config["data"]["training_data_file_id"]
    training_data = load_training_data(config, prediction_cols, train_data_id)

    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]

    # Train and save several models
    for vectorizer_type in vectorizer_types:
        for classifier_type in classifier_types:
            print(f"Training for {vectorizer_type} + {classifier_type} ...")

            if not split_seed:
                split_seed = np.random.randint(100)  # Pick randomly if not given
            grant_tagger = GrantTagger(
                test_size=test_size,
                relevant_sample_ratio=relevant_sample_ratio,
                split_seed=split_seed,
                vectorizer_type=vectorizer_type,
                classifier_type=classifier_type,
                prediction_cols=prediction_cols,
                label_name=label_name,
            )
            X_vect, y = grant_tagger.fit_transform(training_data, train_data_id)

            X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)

            grant_tagger.fit(X_train, y_train)

            grant_info = grant_tagger.train_test_info()

            train_scores = grant_tagger.evaluate(X_train, y_train, extra_scores=True)
            test_scores = grant_tagger.evaluate(X_test, y_test, extra_scores=True)

            evaluation_results = {
                "Train model config": args.config_path,
                "Train scores": train_scores,
                "Test scores": test_scores,
            }

            outout_name = "_".join([vectorizer_type, classifier_type, config_version])
            grant_tagger.save_model(
                os.path.join("models", config_version, outout_name),
                evaluation_results=evaluation_results,
            )

            with open(
                os.path.join("models", config_version, "training_information.json"), "a"
            ) as f:
                f.write(json.dumps({outout_name: grant_info}))
                f.write('\n')


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="configs/train_model/2021.03.16.ini",
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    train_several_models(config)

