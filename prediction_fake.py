import pickle
from keras.models import load_model
from keras.utils import pad_sequences


# 토큰화 방법 불러오기
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 모델 불러오기
model = load_model('my_model.h5')

# 뉴스 내용 입력
news_content = ''' Wed 19 Apr 2017 Aleppo bomb blast kills six Syrian state TV. A bomb blast killed six people and injured 32 in the Salah al-Din district of Aleppo Syrian state television reported on Wednesday without giving further details. Salah al-Din is located west of Aleppos Old City in a district that was part of the last rebel enclave there until it was taken over by the Syrian army in an advance in December. --Reuters 
'''

# 뉴스 내용을 토큰화하고 패딩
sequences = tokenizer.texts_to_sequences([news_content])
MAX_LEN = 512
data = pad_sequences(sequences, maxlen=MAX_LEN)

# 예측 수행
prediction = model.predict(data)

# 결과 출력
if prediction >= 0.5:
    print("The news is predicted to be true.")
else:
    print("The news is predicted to be fake.")

