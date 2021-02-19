"""
This is inspired from textcat.teach

prodigy textcat.teach-tech tech_grants 
    data/prodigy/grants_data.jsonl 
    -F nutrition_labels/prodigy_textcat_teach.py --label 'Tech grant','Not tech grant'
"""

import json
import random
import itertools
from datetime import datetime

import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.db import connect
from prodigy.components.sorters import prefer_low_scores, prefer_high_scores, prefer_uncertain
from prodigy.util import split_string
from typing import List, Optional

def get_jsonl_x_y(file_dir, cat2bin):
    """
    Load the X and y for the model from a JSONL file
    Use cat2bin to convert categories used for tagging into 0 and 1
    """
    X = []
    y = []
    with open(file_dir, 'r') as json_file:
        for json_str in list(json_file):
            data = json.loads(json_str)
            X.append(data['text'])
            y.append(cat2bin[data['label']])

    return X, y

def get_prodigy_x_y(data, cat2bin):
    """
    Get the X and y for the model from a Prodigy dataset
    Use cat2bin to convert categories used for tagging into 0 and 1
    If the answer is reject then use the other tag, 
    e.g. reject 'Not tech grant' means accept 'tech grant'
    """
    data = [eg for eg in data if eg["answer"] != "ignore"]
    X = [annotation['text'] for annotation in data]
    y = []
    for annotation in data:
        label = cat2bin[annotation['label']]
        if annotation['answer']=='accept':
            y.append(label)
        else:
            # We've already filtered about 'ignore'
            # If label=1, append 0 
            # if label=0, append 1
            y.append(abs(label - 1))

    return X, y

class LogRegTFIDF(object):
    def __init__(self, dataset, session_name, label=['Tech grant', 'Not tech grant']):

        self.bin2cat = {0: 'Not tech grant', 1: 'Tech grant'}
        self.cat2bin = {'Not tech grant': 0, 'Tech grant': 1}
 
        self.session_name = session_name
        self.label = label
        
        # Load the current training dataset
        db = connect()
        dataset_examples = db.get_dataset(dataset)
        self.training_X, self.training_y = get_prodigy_x_y(dataset_examples, self.cat2bin)

        # Load the original test data
        self.test_X, self.test_y = get_jsonl_x_y('data/prodigy/existing_test_data.jsonl', self.cat2bin)

        # Train vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
            ngram_range=(1, 2)
            )
        train_X_vect = self.vectorizer.fit_transform(self.training_X)
        test_X_vec = self.vectorizer.transform(self.test_X)

        # Train model
        self.model = LogisticRegression(max_iter=1000)
        self.model = self.model.fit(train_X_vect, self.training_y)

        # Get the beginning test score, before adding new data points
        test_y_pred = self.model.predict(test_X_vec)
        self.test_f1 = f1_score(self.test_y, test_y_pred, average='weighted')

    def __call__(self, stream):
        """
        For each example in the stream use the model to predict
        a label and get the score.
        If the label has a very low probability score, then that means
        it has a high probability of the other label.
        Only output the results when the label is in the self.label list
        """
        
        for eg in stream:
            stream_prob = self.model.predict_proba(self.vectorizer.transform([eg['text']]))[0]
            eg["label"] = self.bin2cat[stream_prob.argmax()]
            if eg["label"] in self.label:
                eg["session_name"] = self.session_name
                score = stream_prob[stream_prob.argmax()]
                yield (score, eg)

    def update(self, examples):
        """
        Rebuild and train the vectorizer and model everytime
        Prodigy updates. Calculate the text F1 and print out
        some train/test scores.
        """

        batch_X, batch_y = get_prodigy_x_y(examples, self.cat2bin)

        if len(batch_X) != 0:
            # Update if the 
            self.training_X = self.training_X + batch_X
            self.training_y = self.training_y + batch_y

            # Refit with collated old training data with new
            self.vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b',
                ngram_range=(1, 2)
                )
            train_X_vect = self.vectorizer.fit_transform(self.training_X)
            
            self.model = LogisticRegression(max_iter=1000)
            self.model = self.model.fit(train_X_vect, self.training_y)

            new_y_pred = self.model.predict(train_X_vect)
            test_y_pred = self.model.predict(self.vectorizer.transform(self.test_X))

            train_f1 = f1_score(self.training_y, new_y_pred, average='weighted')
            self.test_f1 = f1_score(self.test_y, test_y_pred, average='weighted')
            print(f"Training F1: {round(train_f1, 3)}")
            print(f"Test F1: {round(self.test_f1, 3)}")
            print("Train classification report:")
            print(classification_report(self.training_y, new_y_pred))
            print("Test classification report:")
            print(classification_report(self.test_y, test_y_pred))
            print("Test confusion:")
            print(confusion_matrix(self.test_y, test_y_pred))

    def get_progress(self, *args, **kwargs):
        """
        Set the progress bar to be the test F1 score
        """
        return self.test_f1

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "textcat.teach-tech",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    session_name=("A name for this annotation session, can be anything", "option", "s", str),
    sorter=("Which Prodigy stream sorting algorithm to use", "option", "p", str)
)
def textcat_teach(
    dataset: str,
    source: str,
    label: Optional[List[str]] = None,
    session_name: Optional[str] = None,
    sorter: Optional[str] = None
):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    if not session_name:
        session_name = datetime.now().strftime('%y%m%d:%H%M')

    model = LogRegTFIDF(dataset, session_name, label)
    stream = JSONL(source)              # load the data
    stream = model(stream)              # call custom predict function

    # Prodigy's prefer_uncertain looks for scores around 0.5 
    # and assumes scores to be in the 0-1 range, but in our
    # model scores are in the range 0.5-1
    if sorter == 'prefer_high_scores':
        stream = prefer_high_scores(stream)   # sort to prefer high scores
    elif sorter == 'prefer_low_scores':
        stream = prefer_low_scores(stream)   # sort to prefer low scores
    else:
        stream = prefer_uncertain(stream)   # sort to prefer uncertain scores

    return {
        "dataset": dataset,          # dataset to save annotations to
        "stream": stream,            # the incoming stream of examples
        "update": model.update,      # the update callback
        "view_id": "classification",  # annotation interface to use
        "progress": model.get_progress
    }
