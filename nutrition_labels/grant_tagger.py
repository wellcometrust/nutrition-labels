from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from wellcomeml.ml.bert_vectorizer import BertVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from random import sample
from random import seed
from bs4 import BeautifulSoup
import re

from nutrition_labels.useful_functions import pretty_confusion_matrix


class GrantTagger():
    def __init__(
        self,
        ngram_range=(1,2),
        test_size=0.25,
        vectorizer_type='count',
        model_type ='naive_bayes',
        bert_type = 'bert'
        ):
        self.ngram_range = ngram_range
        self.test_size = test_size
        self.vectorizer_type = vectorizer_type
        self.model_type = model_type
        self.bert_type = bert_type
    def transform(self, data):

        self.X = data['Description'].tolist()
        y = data['Relevance code']

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
            self.vectorizer = BertVectorizer(pretrained=self.bert_type)
        else:
            print('Vectorizer type not recognised')
        X_vect = self.vectorizer.fit_transform(self.X)

        if self.vectorizer_type == 'bert' and self.model_type == 'naive_bayes':
            scaler = MinMaxScaler()
            X_vect = scaler.fit_transform(X_vect)

        return X_vect, y

    def split_data(self, X_vect,y, sample_not_relevant, irrelevant_sample_seed,split_seed):
        self.sample_not_relevant = sample_not_relevant

        relevant_sample_index = [ind for ind,x in enumerate(y) if x != 0]
        irrelevant_sample_index = [ind for ind, x in enumerate(y) if x == 0]
        irrelevant_sample_size = int(round((len(relevant_sample_index) * self.sample_not_relevant)))

        if not self.sample_not_relevant and self.sample_not_relevant != 0 :
            # If you don't specify sample_not_relevant
            # then use all the not relevant data points
            sample_size = len(irrelevant_sample_index)
        else:
            # If the inputted sample size is larger than the number of
            # data points, then just use all the data points
            sample_size = min(len(irrelevant_sample_index), irrelevant_sample_size)

        if sample_size < len(irrelevant_sample_index):
            seed(irrelevant_sample_seed)
            sample_index = relevant_sample_index + sample(irrelevant_sample_index,sample_size)
            y = y[sample_index]
            if self.vectorizer_type == 'bert':
                X_vect = [X_vect[i] for i in sample_index]
            else:
                X_vect = X_vect[sample_index]
        # resetting index to remove index from non-sampled data

        X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=self.test_size,
                                                            random_state=split_seed)
        self.train_indices = y_train.index.to_list()
        y_train = y_train.to_list()
        self.test_indices = y_test.index.to_list()
        y_test = y_test.to_list()
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        if self.model_type == 'naive_bayes':
            model = MultinomialNB()
        elif self.model_type == 'SVM':
            model = SVC()
        elif self.model_type == 'log_reg':
            model = LogisticRegression()
        else:
            print('Model type not recognised')
        self.model = model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, print_results=True, average='binary'):

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

        return scores

    def return_mislabeled_data(self, y_actual, y_pred, X_indices):
        X_text = [self.X[i] for i in X_indices]
        X_text_df = pd.DataFrame({'Description': X_text,
                                'True_label':y_actual,
                                'Predicted_label':y_pred})
        return X_text_df

def grant_tagger_experiment(
        sample_not_relevent =1,
        vectorizer_type = 'count',
        model_type ='naive_bayes'
        ):

    grant_tagger = GrantTagger(
        ngram_range=(1, 2),
        test_size=0.25,
        vectorizer_type= vectorizer_type,
        model_type=model_type
    )
    X_vect, y = grant_tagger.transform(data)
    X_train, X_test, y_train, y_test = grant_tagger.split_data(
        X_vect,
        y,
        sample_not_relevant = 1,
        irrelevant_sample_seed = 4,
        split_seed=4)

    grant_tagger.fit(X_train, y_train)
    print('\nNot relevent sample size: ' + str(sample_not_relevent))
    print('\nVectorizer type: ' + vectorizer_type)
    print('\nModel type: ' + model_type)
    print("\nEvaluate training data")
    grant_tagger.evaluate(X_train, y_train)
    print("\nEvaluate test data")
    grant_tagger.evaluate(X_test, y_test)
    print('\nTraining descriptions')
    print(grant_tagger.return_mislabeled_data(y_train, grant_tagger.predict(X_train), grant_tagger.train_indices))
    print('\nTest description')
    test_descriptions = grant_tagger.return_mislabeled_data(y_test,
                                                            grant_tagger.predict(X_test),
                                                            grant_tagger.test_indices)
    print(test_descriptions)
    print("\nMislabled Grant descriptions")
    print(test_descriptions[test_descriptions['True_label'] != test_descriptions['Predicted_label']])


if __name__ == '__main__':

    data = pd.read_csv('data/processed/training_data.csv')

    grant_tagger_experiment(vectorizer_type='bert')
    grant_tagger_experiment(vectorizer_type='bert',model_type='SVM')
    grant_tagger_experiment(vectorizer_type='bert',model_type='log_reg')

