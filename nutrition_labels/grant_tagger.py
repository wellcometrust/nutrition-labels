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
        bert_type = 'bert',
        relevant_sample_ratio = 1
        ):
        self.ngram_range = ngram_range
        self.test_size = test_size
        self.vectorizer_type = vectorizer_type
        self.model_type = model_type
        self.bert_type = bert_type
        self.relevant_sample_ratio = relevant_sample_ratio

    def transform(self, data):

        data['Grant texts'] = data[['Title', 'Grant Programme:Title', 'Description']].agg(
            '. '.join, axis=1
            )
        self.X = data['Grant texts'].tolist()
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

    def split_data(self, X_vect, y, irrelevant_sample_seed, split_seed):

        relevant_sample_index = [ind for ind,x in enumerate(y) if x != 0]
        irrelevant_sample_index = [ind for ind, x in enumerate(y) if x == 0]
        sample_size = int(round(len(relevant_sample_index) * self.relevant_sample_ratio))

        if sample_size < len(irrelevant_sample_index):
            seed(irrelevant_sample_seed)
            sample_index = relevant_sample_index + sample(irrelevant_sample_index, sample_size)
            y = y[sample_index]
            if self.vectorizer_type == 'bert':
                X_vect = [X_vect[i] for i in sample_index]
            else:
                X_vect = X_vect[sample_index]

        y = y.to_list()
        X_train, X_test, y_train, y_test = train_test_split(
            X_vect,
            y,
            test_size=self.test_size,
            random_state=split_seed,
            )

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

        return scores, classification_report(y, y_predict), pretty_confusion_matrix(y, y_predict)

    def return_mislabeled_data(self, y_actual, y_pred, X_indices):
        X_text = [self.X[i] for i in X_indices]
        X_text_df = pd.DataFrame({'Description': X_text,
                                'True_label':y_actual,
                                'Predicted_label':y_pred})
        return X_text_df

    def save_model(self, output_path, split_info, evaluation_results=None, evaluation_results_runs=None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, 'model.pickle'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(output_path, 'vectorizer.pickle'), 'wb') as f:
            pickle.dump(self.vectorizer, f)

        if evaluation_results:
            evaluation_results.update(split_info)
            with open(os.path.join(output_path, 'training_information.txt'), 'w') as f:
                f.write('ngram_range: ' + str(self.ngram_range))
                f.write('\ntest_size: ' + str(self.test_size))
                f.write('\nVectorizer type: ' + self.vectorizer_type)
                f.write('\nModel type: ' + self.model_type)
                f.write('\nBert type (if relevant): ' + str(self.bert_type))
                for key, value in evaluation_results.items():
                    f.write('\n' + key + ': ' + str(value))
        if evaluation_results_runs:
            with open(os.path.join(output_path, 'repeated_results.txt'), 'w') as f:
                for line in evaluation_results_runs:
                    f.write(str(line) + '\n')

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
        default='data/processed/training_data/200807/training_data.csv'
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
    parser.add_argument(
        '--best_of_n',
        help="Specify this and the model will retrain best_of_n times with a \
        different random seed for train_test_split, the model saved will be \
        the best test F1 score and the results of all best_of_n will also be outputted",
        default=None,
        type=int
    )

    return parser

if __name__ == '__main__':

    parser = create_argparser()
    args = parser.parse_args()

    datestamp = datetime.now().date().strftime('%y%m%d')

    data = pd.read_csv(args.training_data_file)
   
    best_test_f1 = 0
    evaluation_results_runs = []

    if not args.best_of_n:
        args.best_of_n = 1
    for run in range(args.best_of_n):
        print(run)
        grant_tagger = GrantTagger(
            ngram_range=(1, 2),
            test_size=0.25,
            vectorizer_type=args.vectorizer_type,
            model_type=args.model_type,
            bert_type=args.bert_type,
            relevant_sample_ratio=args.relevant_sample_ratio
        )

        X_vect, y = grant_tagger.transform(data)
        split_seed = 4 + run
        irrelevant_sample_seed = 4
        X_train, X_test, y_train, y_test = grant_tagger.split_data(
            X_vect,
            y,
            irrelevant_sample_seed=irrelevant_sample_seed,
            split_seed=split_seed)
        split_info = {
            'Split random seed': split_seed,
            'Training data directory': args.training_data_file,
            'Irrelevant sample seed': irrelevant_sample_seed,
            'Relevant sample ratio': args.relevant_sample_ratio
            }
        grant_tagger.fit(X_train, y_train)

        train_scores, _, _ = grant_tagger.evaluate(X_train, y_train, print_results=True)
        test_scores, test_class_rep, test_conf_mat = grant_tagger.evaluate(X_test, y_test, print_results=True)
        evaluation_results = {
            'Train scores': train_scores,
            'Test scores': test_scores,
            'Train size': len(y_train),
            'Test size': len(y_test),
            'Test classification report': test_class_rep,
            'Test confusion matrix': test_conf_mat
            }
        if test_scores['f1'] >= best_test_f1:
            best_test_f1 = test_scores['f1']
            best_grant_tagger = grant_tagger
            best_evaluation_results = evaluation_results
            best_split_info = split_info
        evaluation_results_runs.append(test_scores)
    
    outout_name = '_'.join(
        [args.vectorizer_type, args.model_type]
        )
    if args.bert_type and args.vectorizer_type=='bert':
        outout_name = '_'.join([outout_name, args.bert_type])
    
    outout_name = '_'.join([outout_name, datestamp])

    best_grant_tagger.save_model(
        os.path.join('models', outout_name),
        split_info=best_split_info,
        evaluation_results=best_evaluation_results,
        evaluation_results_runs=evaluation_results_runs
        )
