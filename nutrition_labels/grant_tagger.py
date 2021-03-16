from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
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

from nutrition_labels.useful_functions import pretty_confusion_matrix


class GrantTagger():
    def __init__(
        self,
        ngram_range=(1,2),
        test_size=0.25,
        relevant_sample_ratio=1,
        split_seed=1,
        vectorizer_type='count',
        classifier_type='naive_bayes',
        prediction_cols=['Title', 'Grant Programme:Title', 'Description'],
        label_name="Relevance code"
        ):
        self.ngram_range = ngram_range
        self.test_size = test_size
        self.relevant_sample_ratio = relevant_sample_ratio
        self.split_seed = split_seed
        self.vectorizer_type = vectorizer_type
        self.classifier_type = classifier_type
        self.prediction_cols = *prediction_cols,
        self.label_name = label_name

    def transform(self, data):

        if 'Grant texts' not in data:
            # If the training data hasn't come through Prodigy tagging then this won't exist
            data['Grant texts'] = data[list(self.prediction_cols)].agg(
                '. '.join, axis=1
                )
        self.X = data['Grant texts'].tolist()
        y = data[self.label_name]

        if self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b',
                ngram_range=self.ngram_range
                )
        elif self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b',
                ngram_range=self.ngram_range
                )
        elif self.vectorizer_type == 'bert':
            self.vectorizer = BertVectorizer(pretrained=self.vectorizer_type)
        else:
            print('Vectorizer type not recognised')
        X_vect = self.vectorizer.fit_transform(self.X)

        if 'bert' in self.vectorizer_type and self.classifier_type == 'naive_bayes':
            scaler = MinMaxScaler()
            X_vect = scaler.fit_transform(X_vect)

        return X_vect, y


    def split_data(self, X_vect, y):

        y = y.tolist()
        # Randomly shuffle the data
        random_index = list(range(len(y)))
        seed(self.split_seed)
        shuffle(random_index)
        if self.vectorizer_type == 'bert':
            X_vect = [X_vect[i] for i in random_index]
        else:
            X_vect = X_vect[random_index]
        y = [y[i] for i in random_index]

        relevant_sample_index = [ind for ind, x in enumerate(y) if x != 0]
        irrelevant_sample_index = [ind for ind, x in enumerate(y) if x == 0]
        sample_size = int(round(len(relevant_sample_index) * self.relevant_sample_ratio))
        if sample_size < len(irrelevant_sample_index):
            # Take sample_size equally spaced irrelevant points
            idx = np.round(np.linspace(0, len(irrelevant_sample_index) - 1, sample_size)).astype(int)
            sample_index = relevant_sample_index + [irrelevant_sample_index[i] for i in idx]
            # Make sure they are in order otherwise it'll be all 1111s and then 0000s
            sample_index.sort()
            y = [y[i] for i in sample_index]
            if self.vectorizer_type == 'bert':
                X_vect = [X_vect[i] for i in sample_index]
            else:
                X_vect = X_vect[sample_index]

        # Randomly shuffled at the beginning so turn this off
        X_train, X_test, y_train, y_test = train_test_split(
            X_vect,
            y,
            test_size=self.test_size,
            shuffle=False
            )

        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        if self.classifier_type == 'naive_bayes':
            model = MultinomialNB()
        elif self.classifier_type == 'SVM':
            model = SVC()
        elif self.classifier_type == 'log_reg':
            model = LogisticRegression(max_iter=1000)
        else:
            print('Model type not recognised')
        self.model = model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, print_results=False, average='binary'):

        y_predict = self.model.predict(X)

        scores = {
            'accuracy': accuracy_score(y, y_predict),
            'f1': f1_score(y, y_predict, average=average),
            'precision_score': precision_score(y, y_predict, zero_division=0, average=average),
            'recall_score': recall_score(y, y_predict, zero_division=0, average=average)}

        if print_results:
            print(scores)
            print(classification_report(y, y_predict))
            print(pretty_confusion_matrix(y, y_predict))

        return scores, classification_report(y, y_predict), pretty_confusion_matrix(y, y_predict)

    def save_model(self, output_path, evaluation_results=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(output_path, 'vectorizer.pickle'), 'wb') as f:
            pickle.dump(self.vectorizer, f)

        if evaluation_results:
            evaluation_results['Vectorizer type'] = self.vectorizer_type
            evaluation_results['Classifier type'] = self.classifier_type
            with open(os.path.join(output_path, 'training_information.txt'), 'w') as f:
                for key, value in evaluation_results.items():
                    f.write('\n' + key + ': ' + str(value))

    def load_model(self, output_path):
        with open(os.path.join(output_path, 'model.pickle'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(output_path, 'vectorizer.pickle'), 'rb') as f:
            self.vectorizer = pickle.load(f)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        "--config_path",
        help="Path to config file",
        default="configs/train_model/2021.03.16.ini",
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    # datestamp = datetime.now().date().strftime('%y%m%d')

    vectorizer_types = ast.literal_eval(config["models"]["vectorizer_types"])
    classifier_types = ast.literal_eval(config["models"]["classifier_types"])

    relevant_sample_ratio = config.getfloat("params", "relevant_sample_ratio")
    test_size = config.getfloat("params", "test_size")
    split_seed = config.getint("params", "split_seed")

    prediction_cols = ast.literal_eval(config["data"]["prediction_cols"])
    label_name = config["data"]["label_name"]
    train_data_id = config["data"]["training_data_file_id"]

    training_data = pd.read_csv(config["data"]["training_data_file"])

    if "grants_text_data_file" in config["data"]:
        # Merge training data with another dataset containing the
        # text columns for predicting on
        grants_data = pd.read_csv(config["data"]["grants_text_data_file"])
        right_id = config["data"]["grants_text_data_file_id"]
        training_data = pd.merge(
            training_data,
            grants_data.drop_duplicates(subset='Internal ID')[prediction_cols+[right_id]],
            how='left',
            left_on=train_data_id,
            right_on=right_id)

    # Output the data to a dated folder using the config version date
    # but convert this from 2020.08.07 -> 200807
    config_version = "".join(config["DEFAULT"]["version"].split("."))[2:]

    #!!!!!! DELETE!!!!!
    vectorizer_types = ['count', 'tfidf']
    classifier_types = ['naive_bayes', 'log_reg']
    #!!!!!! !!!!!
    
    for vectorizer_type in vectorizer_types:
        for classifier_type in classifier_types:
            print(f"Training for {vectorizer_type} + {classifier_type} ...")

            grant_tagger = GrantTagger(
                test_size=test_size,
                relevant_sample_ratio=relevant_sample_ratio,
                split_seed=split_seed,
                vectorizer_type=vectorizer_type,
                classifier_type=classifier_type,
                prediction_cols=prediction_cols,
                label_name=label_name
            )
            X_vect, y = grant_tagger.transform(training_data)

            X_train, X_test, y_train, y_test = grant_tagger.split_data(X_vect, y)

            grant_tagger.fit(X_train, y_train)

            train_scores, _, _ = grant_tagger.evaluate(X_train, y_train)
            test_scores, test_class_rep, test_conf_mat = grant_tagger.evaluate(X_test, y_test)
            evaluation_results = {
                'Train model config': args.config_path,
                'Train scores': train_scores,
                'Test scores': test_scores,
                'Train size': len(y_train),
                'Test size': len(y_test),
                'Test classification report': test_class_rep,
                'Test confusion matrix': test_conf_mat
                }

            outout_name = '_'.join(
                [vectorizer_type, classifier_type, config_version]
                )
            grant_tagger.save_model(
                os.path.join('models', config_version, outout_name),
                evaluation_results=evaluation_results
                )

    # Output train/test split used for these models
    # grant code - label - train/test
    grant_tagger = GrantTagger(
                test_size=test_size,
                relevant_sample_ratio=relevant_sample_ratio,
                split_seed=split_seed
            )

    X_train, X_test, y_train, y_test = grant_tagger.split_data(
        training_data[train_data_id],
        training_data[label_name].astype(int)
        )
    train_split = pd.concat([
        pd.DataFrame({train_data_id: X_train, label_name: y_train, 'Train/test': ['Train']*len(y_train)}),
        pd.DataFrame({train_data_id: X_test, label_name: y_test, 'Train/test': ['Test']*len(y_test)})
        ])
    train_split.to_csv(os.path.join('models', config_version, 'test_train_split.csv'), index=False)
