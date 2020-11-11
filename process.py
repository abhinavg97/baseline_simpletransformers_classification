import pandas as pd

# d = pd.read_csv('q/dev1.csv', sep=',', index_col='tweet_id')

# labels = d['labels']

# labels = list(map(lambda x: [0, 1] if x == 1 else [1, 0], labels))

# d['labels'] = labels

# # d.drop(['label'], axis=1, inplace=True)

# d.to_csv('q/final_val.csv', index=True)

d = pd.read_csv('smerp/smerp17_test.csv', index_col=0)

l0 = d['0'].tolist()
l1 = d['1'].tolist()
l2 = d['2'].tolist()
l3 = d['3'].tolist()

d.drop(['0', '1', '2', '3'], axis=1, inplace=True)

d['labels'] = list(map(lambda x1, x2, x3, x4: [x1, x2, x3, x4], l0, l1, l2, l3))

d.to_csv('smerp/smerp_test.csv', index_label='tweet_id')
