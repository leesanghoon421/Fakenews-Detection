import pickle
from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np

# 토큰화 방법 불러오기
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 모델 불러오기
model = load_model('my_model.h5')

#https://news.yahoo.com/head-rt-calls-lindsey-graham-224716821.html
# 뉴스 내용 입력
news_content = '''
Four people were injured in a "massive strike" on the town of Shebekino in Russia's southwestern Belgorod region, Gov. Vyacheslav Gladkov said on Telegram. 

"Two were promptly taken to a hospital in Belgorod," Gladkov said. "The man has shrapnel wounds to the neck and back, the condition is serious, the woman has shrapnel wounds to the arm and forearm. Doctors are now conducting all the necessary examinations."
Gladkov earlier reported that one woman was injured in shelling of the region, which borders northeastern Ukraine

Eight apartment buildings, four homes, a school and two administrative buildings were damaged in the shelling, Gladkov said.

Children will be evacuated from Shebekino and the border town of Grayvoron, with the first 300 taken on Wednesday further east to the town of Voronezh, he added.

On Tuesday, Gladkov reported dozens of strikes by Ukrainian mortar and artillery fire in several areas of Belgorod. One person was killed and two others were injured in an attack on a temporary accommodation center, he said.

CNN cannot independently verify the governor's claims.

It comes after a group of anti-Putin Russian nationals, who are aligned with the Ukrainian army, claimed responsibility for an attack in Belgorod last week. The Ukrainian government distanced itself from the Russian fighters, saying: “In Ukraine these units are part of defense and security forces. In Russia they are acting as independent entities.”
'''

# 뉴스 내용을 토큰화하고 패딩
sequences = tokenizer.texts_to_sequences([news_content])
MAX_LEN = 512
data = pad_sequences(sequences, maxlen=MAX_LEN)

# 예측 수행
prediction = model.predict(data)

# 결과 출력
if prediction >= 0.3:
    print("The news is predicted to be true.")
else:
    print("The news is predicted to be fake.")