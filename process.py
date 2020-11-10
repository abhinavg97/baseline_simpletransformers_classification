import pandas as pd

d = pd.read_csv('q/dev1.csv', sep=',',index_col='tweet_id')

labels = d['labels']

labels = list(map(lambda x: [0,1] if x==1 else [1,0], labels))

d['labels'] = labels

#d.drop(['label'], axis=1, inplace=True)

d.to_csv('q/final_val.csv', index=True)
