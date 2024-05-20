from keras.models import load_model
import pickle
from lime.lime_text import LimeTextExplainer
from keras.utils import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split 
import numpy as np

SEED = 10
df = pd.read_csv('fake_true.csv')
df.dropna(subset = ['text', 'title'], inplace = True)
df['text'] = df['title'] + ' ' + df['text']

X = df['text']
y = df['label']

df['num_words'] = df['text'].apply(lambda x: len(x.split()))


#split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = SEED)

#define Keras Tokenizer
tok = Tokenizer()
tok.fit_on_texts(X_train)

#return sequences
sequences = tok.texts_to_sequences(X_train)
test_sequences = tok.texts_to_sequences(X_test)

#maximum sequence length (512 to prevent memory issues and speed up computation)
MAX_LEN = 512

#padded sequences
X_train_seq = pad_sequences(sequences,maxlen=MAX_LEN)
X_test_seq = pad_sequences(test_sequences,maxlen=MAX_LEN)

# 모델과 토크나이저를 로드합니다.
model = load_model('my_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

# 텍스트 데이터에 대한 설명자를 생성합니다.
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

# 모델의 예측 함수를 정의합니다.
predict_fn = lambda x: np.hstack((1 - model.predict(pad_sequences(tok.texts_to_sequences(x), maxlen=512)), 
                                  model.predict(pad_sequences(tok.texts_to_sequences(x), maxlen=512))))


# 5개의 텍스트 인스턴스에 대해 예측을 설명합니다.
for idx in range(5):
    text_instance = X_test.iloc[idx]
    exp = explainer.explain_instance(text_instance, predict_fn, num_features=15)

    print('Document id: %d' % idx)
    print('Predicted class =', 'Real' if model.predict(pad_sequences(tok.texts_to_sequences([text_instance]), maxlen=512))[0] > 0.5 else 'Fake')
    print('True class: %s' % ('Real' if y_test.iloc[idx] else 'Fake'))
    print('Top contributing words:')
    for word, weight in exp.as_list():
        print('\t', word, '=', weight)
    print("\n---\n")