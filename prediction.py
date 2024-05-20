import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import load_model
from sklearn.metrics import confusion_matrix

import pickle

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
SEED = 10

# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

# Load the saved model
model = load_model('my_model.h5')

# Load the new data for prediction
df_new = pd.read_csv('test_data.csv')
df_new.dropna(subset=['text', 'title'], inplace=True)
df_new['text'] = df_new['title'] + ' ' + df_new['text']

# Reduce the data size to 1/10
df_new = df_new.sample(frac=0.1, random_state=SEED)

X_new = df_new['text']
y_new = df_new['class']

# Preprocess the new data
MAX_LEN=512
sequences_new = tok.texts_to_sequences(X_new)
X_new_seq = pad_sequences(sequences_new, maxlen=MAX_LEN)

# Perform prediction on the new data
y_pred = model.predict(X_new_seq)

# Convert predictions to labels
y_pred_labels = np.where(y_pred >= 0.5, 'Fake', 'Real')

# Calculate evaluation metrics
confusion = confusion_matrix(y_new, y_pred_labels)
precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1])
recall = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
type1_error = confusion[0, 1] / (confusion[0, 0] + confusion[0, 1])
type2_error = confusion[1, 0] / (confusion[1, 0] + confusion[1, 1])

# Print the evaluation metrics
print("Confusion Matrix:")
print(confusion)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Type 1 Error:", type1_error)
print("Type 2 Error:", type2_error)
