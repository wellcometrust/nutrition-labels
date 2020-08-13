import pandas as pd
from nutrition_labels.grant_tagger import GrantTagger
from random import sample, seed

training_data = pd.read_csv('data/processed/training_data.csv')
all_data = pd.read_csv('data/raw/wellcome-grants-awarded-2005-2019.csv')

all_data2 = pd.merge(all_data,training_data, how = 'outer', on = ['Title','Internal ID'])
all_data2 = all_data2.drop(['Description_y', 'Grant Programme:Title_y'],axis = 1)
all_data2 = all_data2.rename(columns={'Description_x':'Description','Grant Programme:Title_x':'Grant Programme:Title'})
all_data2 = all_data2[all_data2['Description'] != 'Not available'].reset_index(drop=True)
grant_tagger = GrantTagger(
    ngram_range=(1, 2),
    test_size=1,
    vectorizer_type='count',
    model_type='naive_bayes',
    bert_type='bert'
)

X_vect, y = grant_tagger.transform(all_data2)
train_inx = [ind for ind,x in enumerate(y) if str(x) != 'nan']
test_inx = [ind for ind,x in enumerate(y) if str(x) == 'nan']
y = y[train_inx].reset_index(drop = True)
X_train  = X_vect[train_inx]
X_test = X_vect[test_inx]

relevant_sample_ratio = 1.625
relevant_sample_index = [ind for ind,x in enumerate(y) if x != 0]
irrelevant_sample_index = [ind for ind, x in enumerate(y) if x == 0]
sample_size = int(round(len(relevant_sample_index) * relevant_sample_ratio))

seed(4)
sample_index = relevant_sample_index + sample(irrelevant_sample_index,sample_size)
y = y[sample_index]
X_train = X_train[sample_index]


fit = grant_tagger.fit(X_train,y)
out_put = grant_tagger.predict(X_test)
all_data3 = all_data2.drop(train_inx)
all_data3['res'] = out_put

out_list = pd.concat([training_data[training_data['Relevance code'] ==1 ]['Internal ID'],all_data3[all_data3['res'] == 1]['Internal ID']])

out_list = out_list.drop_duplicates()
