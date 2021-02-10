"""
This is inspired from textcat.teach

prodigy textcat.teach-tech tech_grants 
    data/prodigy/grants_data.jsonl 
    -F nutrition_labels/prodigy_textcat_teach.py --label 'Tech grant','Not tech grant'

"""

import json

import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.db import connect
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

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "textcat.teach-tech",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
)
def textcat_teach(
    dataset: str,
    source: str,
    exclude: Optional[List[str]] = None,
):
    """
    Collect the best possible training data for a text classification model
    with the model in the loop. Based on your annotations, Prodigy will decide
    which questions to ask next.
    """
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    bin2cat = {0: 'Not tech grant', 1: 'Tech grant'}
    cat2bin = {'Not tech grant': 0, 'Tech grant': 1}

    # Load the current training dataset
    db = connect()
    dataset_examples = db.get_dataset(dataset)
    training_X, training_y = get_prodigy_x_y(dataset_examples, cat2bin)

    # Load the original test data
    test_X, test_y = get_jsonl_x_y('data/prodigy/existing_test_data.jsonl', cat2bin)

    # Train vectorizer
    vectorizer = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 2)
        )
    train_X_vect = vectorizer.fit_transform(training_X)
    test_X_vec = vectorizer.transform(test_X)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model = model.fit(train_X_vect, training_y)

    # Get the beginning test score, before adding new data points
    y_test_pred = model.predict(test_X_vec)
    test_f1 = f1_score(test_y, y_test_pred, average='weighted')

    def update(examples):
        """
        This function is triggered when Prodigy receives annotations
        Train model with new annotations added
        """
        nonlocal test_f1, model, vectorizer

        print(f"Received {len(examples)} annotations.. adding these to the training data")

        new_X, new_y = get_prodigy_x_y(examples, cat2bin)

        # Refit with collated old training data with new
        vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
            ngram_range=(1, 2)
            )
        X = vectorizer.fit_transform(training_X + new_X)
        y = training_y + new_y

        model = LogisticRegression(max_iter=1000)
        model = model.fit(X, y)

        y_pred = model.predict(X)
        y_test_pred = model.predict(vectorizer.transform(test_X))
        train_f1 = f1_score(y, y_pred, average='weighted')
        test_f1 = f1_score(test_y, y_test_pred, average='weighted')
        print(f"Training F1: {round(train_f1, 3)}")
        print(f"Test F1: {round(test_f1, 3)}")
        print("Train classification report:")
        print(classification_report(y, y_pred))
        print("Test classification report:")
        print(classification_report(test_y, y_test_pred))

    def my_prefer_uncertain(stream, model, vectorizer, bin2cat):
        """
        Output stream will be ordered by most uncertain first
        """

        stream_list = list(stream)
        stream_texts = [s['text'] for s in stream_list]
        stream_probs = model.predict_proba(vectorizer.transform(stream_texts))
        for i, s in enumerate(stream_probs):
            stream_list[i]['label'] = bin2cat[s.argmax()]
            stream_list[i]['score'] = s[s.argmax()]

        # Sort by probability score
        stream_list = sorted(stream_list, key=lambda k: k['score'])

        for s in stream_list[0:5]:
            print(f"{s['score']}: {s['label']}: {s['text'][0:100]}")
        # return as generator object
        stream = (s for s in stream_list)
        return stream

    def get_progress(*args, **kwargs):
        """
        Set the progress bar to be the test F1 score
        """
        return test_f1

    stream = my_prefer_uncertain(stream, model, vectorizer, bin2cat)

    return {
        "view_id": "classification",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "update": update,  # Update callback, called with batch of answers
        "exclude": exclude,  # List of dataset names to exclude
        "progress": get_progress
    }
    