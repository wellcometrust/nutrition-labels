from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,  f1_score, precision_score, recall_score
from wellcomeml.ml.bert_vectorizer import BertVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd

from random import sample, seed
from argparse import ArgumentParser
import pickle
from datetime import datetime
import os

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

    def split_data(self, X_vect, y, relevant_sample_ratio, irrelevant_sample_seed, split_seed):

        self.relevant_sample_ratio = relevant_sample_ratio
        relevant_sample_index = [ind for ind,x in enumerate(y) if x != 0]
        irrelevant_sample_index = [ind for ind, x in enumerate(y) if x == 0]
        sample_size = int(round(len(relevant_sample_index) * relevant_sample_ratio))

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

    def save_model(self, output_path, evaluation_results=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(output_path, 'vectorizer.pickle'), 'wb') as f:
            pickle.dump(self.vectorizer, f)

        if evaluation_results:
            with open(os.path.join(output_path, 'training_information.txt'), 'w') as f:
                f.write('ngram_range: ' + str(self.ngram_range))
                f.write('\ntest_size: ' + str(self.test_size))
                f.write('\nVectorizer type: ' + self.vectorizer_type)
                f.write('\nModel type: ' + self.model_type)
                f.write('\nBert type (if relevant): ' + self.bert_type)
                f.write('\nNot relevent sample size: ' + str(self.relevant_sample_ratio))
                for key, value in evaluation_results.items():
                    f.write('\n' + key + ': ' + str(value))

    def load_model(self, output_path):
        with open(os.path.join(output_path, 'model.pickle'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(output_path, 'vectorizer.pickle'), 'rb') as f:
            self.vectorizer = pickle.load(f)


def create_argparser():

    parser = ArgumentParser()
    parser.add_argument(
        '--training_data_file',
        help='Path to the training data csv',
        default='data/processed/training_data.csv'
    )
    parser.add_argument(
        '--vectorizer_type',
        help="Which vectorizer to use, options are 'count', 'tfidf' or 'bert'",
        default='count'
    )
    parser.add_argument(
        '--relevant_sample_ratio',
        help='There is more not-relevant data in the training data, so this is the ratio \
        of relevant to not-relevant data to sample, e.g. 1 means equal numbers of\
        both, 0.5 means use half the amount of not-relevant compared to relevant',
        default=1,
        type=float
    )
    parser.add_argument(
        '--model_type',
        help="Which model type to use, options are 'naive_bayes', 'SVM' and 'log_reg'",
        default='naive_bayes'
    )
    parser.add_argument(
        '--bert_type',
        help="If you are using the bert vectoriser then which type do you want,\
        options are 'bert' and 'scibert', this doesn't need to be defined if you\
        aren't using bert",
        default=None
    )

    return parser

if __name__ == '__main__':

    parser = create_argparser()
    args = parser.parse_args()

    data = pd.read_csv(args.training_data_file)
   
    grant_tagger = GrantTagger(
        ngram_range=(1, 2),
        test_size=0.25,
        vectorizer_type=args.vectorizer_type,
        model_type=args.model_type,
        bert_type=args.bert_type
    )

    X_vect, y = grant_tagger.transform(data)
    X_train, X_test, y_train, y_test = grant_tagger.split_data(
        X_vect,
        y,
        relevant_sample_ratio=args.relevant_sample_ratio,
        irrelevant_sample_seed=4,
        split_seed=4)

    grant_tagger.fit(X_train, y_train)

    train_scores = grant_tagger.evaluate(X_train, y_train, print_results=True)
    test_scores = grant_tagger.evaluate(X_test, y_test, print_results=True)
    evaluation_results = {
        'Train scores': train_scores,
        'Test scores': test_scores,
        'Train size': len(y_train),
        'Test size': len(y_test)
        }
    
    outout_name = '_'.join(
        [args.vectorizer_type, args.model_type]
        )
    if args.bert_type and args.vectorizer_type=='bert':
        outout_name = '_'.join([outout_name, args.bert_type])
    
    datestamp = datetime.now().date().strftime('%y%m%d')
    outout_name = '_'.join([outout_name, datestamp])

    grant_tagger.save_model(
        os.path.join('models', outout_name),
        evaluation_results=evaluation_results,
        )

    # print('\nTraining descriptions')
    # print(grant_tagger.return_mislabeled_data(y_train, grant_tagger.predict(X_train), grant_tagger.train_indices))
    # print('\nTest description')
    # test_descriptions = grant_tagger.return_mislabeled_data(y_test,
    #                                                         grant_tagger.predict(X_test),
    #                                                         grant_tagger.test_indices)
    # print(test_descriptions)
    # print("\nMislabled Grant descriptions")
    # print(test_descriptions[test_descriptions['True_label'] != test_descriptions['Predicted_label']])

    # # Loading a trained model and vectorizer to predict new data:

    # grant_tagger_loaded = GrantTagger()
    # grant_tagger_loaded.load_model(os.path.join('models', 'count_naive_bayes_200806'))

    # new_grants = [
    #     'Is this a grant about tools and stuff like models',
    #     'in this grant I created a model of the human health care system in the UK. The model is open source and several python software packages have been released.']
    # new_grants_vect = grant_tagger_loaded.vectorizer.transform(new_grants)
    # grant_tagger_loaded.predict(new_grants_vect)

    # # Predict all the training data
    # X_vect = grant_tagger_loaded.vectorizer.transform(data['Description'].tolist())
    # y = data['Relevance code']
    # grant_tagger_loaded.evaluate(X_vect, y, print_results=True, average='binary')

    # # Split the train and test and evaluate
    # X_train, X_test, y_train, y_test = grant_tagger_loaded.split_data(
    #     X_vect,
    #     y,
    #     relevant_sample_ratio=args.relevant_sample_ratio,
    #     irrelevant_sample_seed=4,
    #     split_seed=4)

    # grant_tagger_loaded.evaluate(X_train, y_train, print_results=True)
    # grant_tagger_loaded.evaluate(X_test, y_test, print_results=True)

