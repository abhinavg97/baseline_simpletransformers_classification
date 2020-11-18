import pandas as pd

set1 = 'train'

d = pd.read_csv(f'neq/{set1}.tsv', sep='	', index_col=0)

labels = d['labels'].tolist()

labels = list(map(lambda x: [0, 1] if x == 'relevant' else [1, 0], labels))

d['labels'] = labels


d.to_csv(f'neq/{set1}.csv', index_label='tweet_id')

# d = pd.read_csv('fire/fire16_test.csv', index_col=0)

# l0 = d['0'].tolist()
# l1 = d['1'].tolist()
# l2 = d['2'].tolist()
# l3 = d['3'].tolist()

# d.drop(['0', '1', '2', '3'], axis=1, inplace=True)

# d['labels'] = list(map(lambda x1, x2, x3, x4: [x1, x2, x3, x4], l0, l1, l2, l3))

# d.to_csv('fire/test.csv', index_label='tweet_id')

# df = pd.read_csv('sandy/full.csv', sep=',', index_col=0)
# train_val = df.sample(frac=0.7, random_state=23)
# test = df.drop(train_val.index)


# train = train_val.sample(frac=(6/7.0), random_state=23)

# val = train_val.drop(train.index)

# train.to_csv('sandy/train.csv', index_label='tweet_id')
# val.to_csv('sandy/val.csv', index_label='tweet_id')
# test.to_csv('sandy/test.csv', index_label='tweet_id')
