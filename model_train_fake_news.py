import numpy as np
import pandas as pd
import gensim
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report, accuracy_score
import preprocess_kgptalkie as ps
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow import keras

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

# Analyse des sujets
fake['subject'].value_counts()

plt.figure(figsize=(10, 6))
sns.countplot(x = 'subject', data=fake)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x = 'subject', data=real)
plt.show()


# Wordcloud
text = ' '.join(fake['text'].tolist())
textTrue = ' '.join(real['text'].tolist())
wordcloudFake = WordCloud(width=1920, height=1080).generate(text)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloudFake)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

wordcloudTrue = WordCloud(width=1920, height=1080).generate(textTrue)
fig = plt.figure(figsize=(10,10))
plt.imshow(wordcloudTrue)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

textFake = ''.join(fake['text'].tolist()) # On veut une seule string pour conter les mots les plus souvents utilisés
textTrue = ''.join(real['text'].tolist())

unknown_publishers = []
for index, row in enumerate(real.text.values):
    try:
        record = row.split('-', maxsplit=1)
        assert(len(record[0]) < 120) # Si c'est un tweet
    except:
        unknown_publishers.append(index)

real = real.drop(8970, axis=0) # supprime texte vide

# On obtient la liste des producteurs
publisher = []
tmp_text = [] # temporaire

for index, row in enumerate(real.text.values):
    if index in unknown_publishers:
        tmp_text.append(row)
        publisher.append('Unknown')
    else:
        record = row.split('-', maxsplit=1)
        publisher.append(record[0].strip())
        tmp_text.append(record[1].strip())

# On remplace les données avec les nouvelles données propres
real['publisher'] = publisher
real['text'] = tmp_text

empty_fake_index = [index for index, text in enumerate(fake.text.tolist()) if str(text).strip() == ""]

# On rassemble le titre et le contenu
real['text'] = real['title'] + " " + real['text']
fake['text'] = fake['title'] + " " + fake['text']

# On met tout en minuscules
real['text'] = real['text'].apply(lambda x: str(x).lower())
fake['text'] = fake['text'].apply(lambda x: str(x).lower())

# On assigne un entier à chaque type de brève
real['class'] = 1
fake['class'] = 0

real = real[['text', 'class']]
fake = fake[['text', 'class']]

# On rassemble les deux
data = real.append(fake, ignore_index=True)

# Suppression des caractères spéciaux
data['text'] = data['text'].apply(lambda x: ps.remove_special_chars(x))

y = data['class'].values

X = [d.split() for d in data['text'].tolist()]

dimension = 100 # Dimension de chaque vecteur pour chaque mot entrainé
wordToVector_Model = gensim.models.Word2Vec(sentences=X, size=dimension, window=10, min_count=1)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)

nos = np.array([len(x) for x in X])

# On supprime les phrases de plus de 1000 mots
maxlen = 1000
X = pad_sequences(X, maxlen=maxlen)

vocab_size = len(tokenizer.word_index) + 1
vocab = tokenizer.word_index

def get_weigth_matrix(model):
    weight_matrix = np.zeros((vocab_size, dimension))

    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]

    return weight_matrix

embedding_vectors = get_weigth_matrix(wordToVector_Model)

model = Sequential()
model.add(Embedding(vocab_size, output_dim=dimension, weights=[embedding_vectors], input_length=maxlen, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X,y)

model.fit(X_train , y_train, validation_split=0.3, epochs=6)

y_pred = (model.predict(X_test) >= 0.5).astype(int)

accuracy_score(y_test, y_pred)

x = ["Hong Kong has announced a two-week ban on incoming flights from eight countries and tightened local Covid restrictions as authorities feared a fifth wave of coronavirus in the city.The restrictions were announced as health authorities scoured the city for the contacts of a Covid patient, some of whom had been onboard a Royal Caribbean ship that was ordered to cut short its “cruise to nowhere” and return to port."]
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=maxlen)
y_pred = (model.predict(x) >= 0.5).astype(int)
print(model.predict(x))

x = ["Climate Change is fake.Climate Change is fake.Climate Change is fake.Climate Change is fake.Climate Change is fake.Climate Change is fake.Climate Change is fake.Climate Change is fake.Climate Change is fake."]
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=maxlen)
y_pred = (model.predict(x) >= 0.5).astype(int)
print(model.predict(x))

x = ["In the year since the assault on the Capitol by a pro-Trump mob, more than 700 people have been arrested, with little public indication from the Justice Department of how high the investigation might reach."]
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=maxlen)
y_pred = (model.predict(x) >= 0.5).astype(int)
print(model.predict(x))

model.summary()
model.save('saved_model/fakenewsmodel.h5')

testmodel = keras.models.load_model('saved_model/fakenewsmodel')
testmodel.summary()

y_pred_export = (testmodel.predict(X_test) >= 0.5).astype(int)
print(classification_report(y_test, y_pred_export))

x_phrase = ['Climate change does not exist']
x_phrase = tokenizer.texts_to_sequences(x_phrase)

x_phrase = pad_sequences(x_phrase, maxlen=maxlen)

testmodel.predict(x_phrase)[0][0]

"""# **Plus la prédiction est basse, plus la probabilité que le texte soit une fake news est élevée**"""

x_example = ["Hong Kong has announced a two-week ban on incoming flights from eight countries and tightened local Covid restrictions as authorities feared a fifth wave of coronavirus in the city.The restrictions were announced as health authorities scoured the city for the contacts of a Covid patient, some of whom had been onboard a Royal Caribbean ship that was ordered to cut short its “cruise to nowhere” and return to port."]
x_example = tokenizer.texts_to_sequences(x_example)
x_example = pad_sequences(x_example, maxlen=maxlen)
print(testmodel.predict(x_example))

x_example = ["Covid-19 does not exists and has been invented by China."]
x_example = tokenizer.texts_to_sequences(x_example)
x_example = pad_sequences(x_example, maxlen=maxlen)
print(testmodel.predict(x_example))
