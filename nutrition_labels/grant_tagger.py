from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

    soup = BeautifulSoup(string)
    string_out = soup.get_text()
    string_out = string_out.strip('\n')
    string_out = string_out.strip('\xa0')
    string_out = re.sub('  ','',string_out)
    return(string_out)

class GrantTagger():
    def __init__(self, sample_4s=50, ngram_range=(1,2), test_size=0.25,random_state = 4):
        self.sample_4s = sample_4s
        self.ngram_range = ngram_range
        self.test_size = test_size
        self.random_state = random_state

    def transform(self, data):

        equal_data = data.loc[data['code'] != 5.0]
        equal_data = pd.concat([equal_data,
                                data.loc[data['code'] == 5.0].sample(n = self.sample_4s, random_state= self.random_state)])

        # resetting index to remove index from non-sampled data
        equal_data = equal_data.reset_index(drop = True)

        # Meaningful if 1,2,3 -> reset to 1
        equal_data['code'] = [int(i) for i in (equal_data['code'] != 5.0).tolist()]

        self.X = [remove_useless_string(i) for i in equal_data['Description'].tolist()]
        y = equal_data['code']

        self.vectorizer = CountVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=self.ngram_range)
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

        # model = Pipeline([
        #         ('vectorizer', CountVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b')),
        #         ('bayes', MultinomialNB())
        #     ])

        bayes = MultinomialNB()
        self.model = bayes.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):

        y_predict = self.model.predict(X)

        accuracy = accuracy_score(y, y_predict)

        print(accuracy)
        print(classification_report(y, y_predict))
        print(confusion_matrix(y, y_predict))

    def return_mislabeled_data(self, y_actual, y_pred, X_indices):
        X_text = [self.X[i] for i in X_indices]
        X_text_df = pd.DataFrame({'Description': X_text,
                                'True_label':y_actual,
                                'Predicted_label':y_pred})
        return X_text_df


if __name__ == '__main__':
    
    data = pd.read_csv('data/processed/training_data.csv')

    grant_tagger = GrantTagger(sample_4s=50, ngram_range=(1,2), test_size=0.25, random_state= 4)
    X_train, X_test, y_train, y_test = grant_tagger.transform(data)
    grant_tagger.fit(X_train, y_train)

    print("Evaluate training data")
    grant_tagger.evaluate(X_train, y_train)
    print("Evaluate test data")
    grant_tagger.evaluate(X_test, y_test)
    print('Training descriptions')
    print(grant_tagger.return_mislabeled_data(y_train, grant_tagger.predict(X_train), grant_tagger.train_indices))
    print('Test description')
    test_descriptions = grant_tagger.return_mislabeled_data(y_test,
                                                            grant_tagger.predict(X_test),
                                                            grant_tagger.test_indices)
    print(test_descriptions)
    print("Mislabled Grant descriptions")
    print(test_descriptions[test_descriptions['True_label'] != test_descriptions['Predicted_label']])

######

