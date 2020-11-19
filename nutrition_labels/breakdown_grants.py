import pandas as pd
import re
import matplotlib.pyplot as plt
data = pd.read_csv('data/processed/tool_grants.csv')
data_small = data[['Internal ID','Amount Awarded']]

def short_ids(id):
    new_id = re.sub('/.*','',id)
    if len(new_id) < 6:
        new_id = 0 + new_id
    return new_id

data_small['ID 6'] =  data_small['Internal ID'].apply(short_ids)
data_small['Amount Awarded'] = data_small['Amount Awarded'].apply(lambda x: int(re.sub(',','',x)))
groups = data_small.groupby('ID 6')['Amount Awarded'].sum().reset_index().sort_values('Amount Awarded', ascending= False)
groups = groups[groups['Amount Awarded'] > 0]
groups['cumu money'] = groups['Amount Awarded'].cumsum()
groups['cumu grant'] = range(1,len(groups) + 1)
groups['percent money'] = groups['cumu money']/ sum(groups['Amount Awarded']) * 100
groups['percent grants'] = groups['cumu grant']/len(groups) * 100

plt.scatter(groups['percent grants'],groups['percent money'])

rich_grants = groups[round(groups['percent money']) <= 20 ]
data['ID 6'] =  data['Internal ID'].apply(short_ids)

rich_grants = pd.merge(rich_grants, data[['Internal ID','ID 6','Title','Amount Awarded','Description']], how = 'left', on = 'ID 6')