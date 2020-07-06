from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

import pandas as pd

from bs4 import BeautifulSoup
import re

# getting WT grant number
def remove_useless_string(string):
    '''
    cleans the grant descriptions of artifacts such as <br />
    :param string: description string
    :return: clean string
    '''

    soup = BeautifulSoup(string, features="lxml")
    string_out = soup.get_text()
    string_out = string_out.strip('\n')
    string_out = string_out.strip('\xa0')
    string_out = re.sub('  ','',string_out)
    return(string_out)


def pretty_confusion_matrix(y, y_predict, labels=[0,1]):
    '''
    sklearn's confusion matrix doesn't give informative row and col labels
    Confusion matrix whose i-th row and j-th column entry indicates 
    the number of samples with true label being i-th class and prediced label being j-th class.
    '''

    cm = pd.DataFrame(confusion_matrix(y, y_predict, labels = labels))

    cm.rename(
        index={0:'Actually not relevant', 1:'Actually relevant'}, 
        columns={0:'Predicted not relevant', 1:'Predicted relevant'}, 
        inplace=True)
    return cm

    
class GrantTagger():
    def __init__(self, sample_not_relevant=50, ngram_range=(1,2), test_size=0.25,random_state = 4, vectorizer_type='count'):
        self.sample_not_relevant = sample_not_relevant
        self.ngram_range = ngram_range
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer_type = vectorizer_type

    def transform(self, data):

        equal_data = data.loc[data['code'] != 5.0]
        irrelevant_data = data.loc[data['code'] == 5.0]
        if not self.sample_not_relevant:
            # If you don't specify sample_not_relevant
            # then use all the not relevant data points
            sample_size = len(irrelevant_data)
        else:
            # If the inputted sample size is larger than the number of 
            # data points, then just use all the data points
            sample_size = min(len(irrelevant_data), self.sample_not_relevant)

        equal_data = pd.concat([equal_data,
                                irrelevant_data.sample(n = sample_size, random_state= self.random_state)])

        # resetting index to remove index from non-sampled data
        equal_data = equal_data.reset_index(drop = True)

        # Meaningful if 1,2,3 -> reset to 1
        equal_data['code'] = [int(i) for i in (equal_data['code'] != 5.0).tolist()]

        self.X = [remove_useless_string(i) for i in equal_data['Description'].tolist()]
        y = equal_data['code']

        if self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=self.ngram_range)
        elif self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=self.ngram_range)
        else:
            print('Vectorizer type not recognised')
        X_vect = self.vectorizer.fit_transform(self.X)
        # vectorizer.get_feature_names()
        # X_vect.toarray()
        # word_list = vectorizer.get_feature_names();    
        # count_list = X_vect.toarray().sum(axis=0)    
        # dict(zip(word_list,count_list)))


        X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=self.test_size,
                                                            random_state=42)
        self.train_indices = y_train.index.to_list()
        y_train = y_train.to_list()
        self.test_indices = y_test.index.to_list()
        y_test = y_test.to_list()
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):

        bayes = MultinomialNB()
        self.model = bayes.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, print_results=True):

        y_predict = self.model.predict(X)

        scores = {
            'accuracy': accuracy_score(y, y_predict),
            'f1': f1_score(y, y_predict),
            'precision_score': precision_score(y, y_predict),
            'recall_score': recall_score(y, y_predict)}

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


if __name__ == '__main__':
    
    data = pd.read_csv('data/processed/training_data.csv')

    grant_tagger = GrantTagger(sample_not_relevant=50, ngram_range=(1,2), test_size=0.25, random_state= 4)
    X_train, X_test, y_train, y_test = grant_tagger.transform(data)
    grant_tagger.fit(X_train, y_train)

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

######

