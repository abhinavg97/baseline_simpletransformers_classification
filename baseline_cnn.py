from ast import literal_eval
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import pandas as pd


def read_data():

    train_df = pd.read_csv('smerp/smerp_train.csv', index_col=0)
    # val_df = pd.read_csv('smerp/smerp_val.csv', index_col=0)
    test_df = pd.read_csv('smerp/smerp_test.csv', index_col=0)

    train_df['labels'] = list(map(lambda label_list: literal_eval(label_list), train_df['labels'].tolist()))
    # val_df['labels'] = list(map(lambda label_list: literal_eval(label_list), val_df['labels'].tolist()))
    test_df['labels'] = list(map(lambda label_list: literal_eval(label_list), test_df['labels'].tolist()))

    # train_df['text'] = process_text(train_df)
    # val_df['text'] = process_text(val_df)
    # test_df['text'] = process_text(test_df)

    # train_df.to_csv("smerp/train.csv")
    # val_df.to_csv("q/final_val2")
    # test_df.to_csv("smerp/test.csv")

    return train_df, test_df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Read data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_df, test_df = read_data()

label_id_to_label_text = {0: "l0", 1: "l1", 2: "l2", 3: "l3"}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vect = Tokenizer()
vect.fit_on_texts(train_df['plot_synopsis'])
vocab_size = len(vect.word_index) + 1
print(vocab_size)


encoded_docs_train = vect.texts_to_sequences(train_df['preprocessed_plots'])
max_length = vocab_size
padded_docs_train = pad_sequences(encoded_docs_train, maxlen=1200, padding='post')
print(padded_docs_train)


encoded_docs_test =  vect.texts_to_sequences(test['preprocessed_plots'])
padded_docs_test = pad_sequences(encoded_docs_test, maxlen=1200, padding='post')
encoded_docs_cv = vect.texts_to_sequences(cv['preprocessed_plots'])
padded_docs_cv = pad_sequences(encoded_docs_cv, maxlen=1200, padding='post')

model = Sequential()
# Configuring the parameters
model.add(Embedding(vocab_size, output_dim=50, input_length=1200))
model.add(LSTM(128, return_sequences=True))  
# Adding a dropout layer
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
# Adding a dense output layer with sigmoid activation
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy')
history = model.fit(padded_docs_train, y_train,
                    class_weight='balanced',
                    epochs=5,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[])

predictions=model.predict([padded_docs_test])