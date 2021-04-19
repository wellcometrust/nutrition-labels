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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from wellcomeml.ml.bert_vectorizer import BertVectorizer
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
    threshold : float (default None)
        A prediction probability threshold that needs to be satisfied for a datapoint
        to be predicted as tech.

    Methods
    -------
    transform(data)
        Vectorize text data using the fitted vectorizer
    split_data(data, train_data_id)
        Split the data into train and test sets
    fit(X, y)
        Fit the model - fit a vectorizer using the text data, and train the classifier
    predict(X)
        Make predictions using the trained model
    predict_proba(X)
        Output prediction probabilities using the trained model
    evaluate(X, y, extra_scores, average)
        Evaluate various metrics using the model.
    train_test_info(train_ids, y_train, test_data, unseen_data)
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
        threshold=None,
    ):
        self.test_size = test_size
        self.relevant_sample_ratio = relevant_sample_ratio
        self.split_seed = split_seed
        self.vectorizer_type = vectorizer_type
        self.classifier_type = classifier_type
        self.prediction_cols = (*prediction_cols,)
        self.label_name = label_name
        self.threshold = threshold

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
            
        return X_vect

    def split_data(self, data, train_data_id="Internal ID"):

        if "Grant texts" not in data:
            # If the training data hasn't come through Prodigy tagging then this won't exist
            data = self.process_grant_text(data)

        data = data.astype({self.label_name: int})
        # The index is important, so make sure its reset to begin with
        data.reset_index(inplace=True, drop=True)

        # Randomly shuffle the data
        random_index = list(range(len(data)))
        seed(self.split_seed)
        shuffle(random_index)
        data = data.iloc[random_index, :]
        data.reset_index(inplace=True)

        relevant_sample_index = data[data[self.label_name] != 0].index.values.tolist()
        irrelevant_sample_index = data[data[self.label_name] == 0].index.values.tolist()

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

            # Store data not in sample
            not_sample_index = data.index.difference(sample_index)
            unused_data = data.iloc[not_sample_index, :]
            X_unused = unused_data['Grant texts'].tolist()
            y_unused = unused_data[self.label_name].tolist()
            unused_ids = unused_data[train_data_id].tolist()
            # Sample data
            data = data.iloc[sample_index, :]
        else:
            X_unused = []
            y_unused = []
            unused_ids = []

        X = data[["Grant texts", train_data_id]]
        y = data[self.label_name].tolist()

        # Randomly shuffled at the beginning so turn this off
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )

        train_ids = X_train[train_data_id].tolist()
        test_ids = X_test[train_data_id].tolist()

        X_train = X_train['Grant texts'].tolist()
        X_test = X_test['Grant texts'].tolist()

        train_data = (X_train, y_train, train_ids)
        test_data = (X_test, y_test, test_ids)
        unseen_data = (X_unused, y_unused, unused_ids)

        return train_data, test_data, unseen_data

    def fit(self, X, y):

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
            if self.classifier_type == "naive_bayes":
                self.vectorizer = Pipeline([('vec', self.vectorizer), ('scaler', MinMaxScaler())])
        else:
            print("Vectorizer type not recognised")

        if self.classifier_type == "naive_bayes":
            model = MultinomialNB()
        elif self.classifier_type == "SVM":
            model = SVC(probability=True)
        elif self.classifier_type == "log_reg":
            model = LogisticRegression(max_iter=1000)
        else:
            print("Model type not recognised")

        self.X_train_vect = self.vectorizer.fit_transform(X)
        self.model = model.fit(self.X_train_vect, y)

    def apply_threshold(self, y_predict, pred_probs):
        return y_predict*(np.max(pred_probs, axis=1) >= self.threshold)

    def predict(self, X):
        y_predict = self.model.predict(X).astype(int)
        if self.threshold:
            # If the prediction probability is over a threshold then allow a 1
            # prediction to stay as 1, otherwise switch to 0.
            # A prediction of 0 will stay at 0 regardless of probability.
            pred_probs = self.model.predict_proba(X)
            y_predict = self.apply_threshold(y_predict, pred_probs)

        return y_predict

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y, extra_scores=False, average="binary"):

        y_predict = self.predict(X)

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

    def train_test_info(self, train_ids, y_train, test_data, unseen_data):
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
        
        (X_test, y_test, test_ids) = test_data
        (X_unused, y_unused, unused_ids) = unseen_data

        X_test_vect = self.vectorizer.transform(X_test)
        X_unused_vect = self.vectorizer.transform(X_unused)

        all_info = {
            'Train': (self.X_train_vect, train_ids, y_train),
            'Test': (X_test_vect, test_ids, y_test),
            'Not used': (X_unused_vect, unused_ids, y_unused)
        }

        training_info = {}
        for split_type, (vectors, grant_ids, actuals) in all_info.items():
            if len(grant_ids) != 0:
                preds = self.predict(vectors)
                pred_probs = self.predict_proba(vectors)
                pred_probs = np.max(pred_probs, axis=1)
                for grant_id, actual, pred, pred_prob in zip(grant_ids, actuals, preds, pred_probs):
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

            train_data, test_data, unseen_data = grant_tagger.split_data(training_data, train_data_id)
            (X_train, y_train, train_ids) = train_data
            (X_test, y_test, _) = test_data

            grant_tagger.fit(X_train, y_train)

            grant_info = grant_tagger.train_test_info(train_ids, y_train, test_data, unseen_data)

            X_test_vect = grant_tagger.vectorizer.transform(X_test)

            train_scores = grant_tagger.evaluate(
                grant_tagger.X_train_vect,
                y_train,
                extra_scores=True
                )
            test_scores = grant_tagger.evaluate(X_test_vect, y_test, extra_scores=True)

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
