import pandas as pd

from datetime import datetime
import os

from nutrition_labels.grant_tagger import GrantTagger


if __name__ == '__main__':
    
    training_data_file = 'data/processed/training_data/200807/training_data.csv'

    datestamp = datetime.now().date().strftime('%y%m%d')

    data = pd.read_csv(training_data_file)

    model_types = ['naive_bayes', 'log_reg', 'SVM']
    vectorizer_types = ['count', 'tfidf', 'bert', 'scibert']

    relevant_sample_ratio = 1
    num_rand_seeds = 10

    for vectorizer_type in vectorizer_types:
        print(vectorizer_type)
        if vectorizer_type in ['bert', 'scibert']:
            bert_type = vectorizer_type
            vectorizer_type = 'bert'
        else:
            bert_type = 'bert'
        for model_type in model_types:
            print(model_type)

            grant_tagger = GrantTagger(
                    ngram_range=(1, 2),
                    test_size=0.25,
                    vectorizer_type=vectorizer_type,
                    model_type=model_type,
                    bert_type=bert_type,
                    relevant_sample_ratio=relevant_sample_ratio
                    )

            X_vect, y = grant_tagger.transform(data)

            evaluation_results_runs = []
            for split_seed in range(num_rand_seeds):
                print(split_seed)
                X_train, X_test, y_train, y_test = grant_tagger.split_data(
                    X_vect,
                    y,
                    split_seed=split_seed)
                
                grant_tagger.fit(X_train, y_train)

                train_scores, _, _ = grant_tagger.evaluate(X_train, y_train, print_results=False)
                test_scores, test_class_rep, test_conf_mat = grant_tagger.evaluate(X_test, y_test, print_results=False)
                
                evaluation_results = {
                    'Split random seed': split_seed,
                    'Training data directory': training_data_file,
                    'Relevant sample ratio': relevant_sample_ratio,
                    'Train scores': train_scores,
                    'Test scores': test_scores,
                    'Train size': len(y_train),
                    'Test size': len(y_test)
                    }

                evaluation_results_runs.append(evaluation_results)

            outout_name = '_'.join(
                [vectorizer_type, model_type]
                )
            if bert_type and vectorizer_type=='bert':
                outout_name = '_'.join([outout_name, bert_type])

            output_path = os.path.join('model_repeats', '_'.join([outout_name, datestamp]))

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            with open(os.path.join(output_path, 'repeated_results.txt'), 'w') as f:
                for line in evaluation_results_runs:
                    f.write(str(line) + '\n')

