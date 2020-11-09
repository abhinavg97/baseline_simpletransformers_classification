import pandas as pd

d = pd.read_csv('q/train.csv', sep='	',index_col='tweet_id')

labels = d['label']

labels = list(map(lambda x: 1 if x=='relevant' else 0, labels))

d['labels'] = labels

d.drop(['label'], axis=1, inplace=True)

d.to_csv('q/train1.csv', index=True)