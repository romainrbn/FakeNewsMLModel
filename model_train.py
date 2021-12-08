import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

# On récupère les données depuis une URL et on l'enregistre sur le systeme pour ne pas le re-dl à chaque fois
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='.')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# Liste des dataset : ['train', 'test', ...]
os.listdir(dataset_dir)

# Liste des fichiers dans le dossier train
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# On print un fichier parmi tous les fichiers dans train
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

# Pour préparer le dataset pour la classification, on a besoin de deux dossiers sur le disque
# Le premier dossier est 'class_a' qui correspond aux avis positif et le deuxieme est 'class_b' qui
# correspond aux avis négatifs

# On supprime d'abord les dossiers inutiles
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#### --- Début apprentissage --- ####
# Il nous manque un fichier de validation, on le crée avec un split 80:20 des données d'entrainement
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,  # taille du split
    subset='training',
    seed=seed
)
''' On a comme output : 
    Found 25000 files belonging to 2 classes.
    Using 20000 files for training.

    On a donc 25,000 exemples dans le dossier d'entrainement
    Et on utilisera 20,000 exemples (80%) pour l'entrainement
'''

# On print quelques données d'exemple avec tf :
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("Label 0 correspond à", raw_train_ds.class_names[0])
print("Label 1 correspond à", raw_train_ds.class_names[1])

# On vérifie que c'est bien lié
''' Output : 
    Label 0 correspond à neg
    Label 1 correspond à pos
'''

# On va ensuite créer un dataset de test et un dataset de validation
# On utilise les 5000 avis restants pour la validation

# Données de validation
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,  # 20% restants
    subset='validation',
    seed=seed
)
''' On a comme output : 
    Found 25000 files belonging to 2 classes.
    Using 5000 files for validation.
'''

# Données de test
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size
)


## Préparation du dataset pour l'entrainement ##
# la fonction dépend du dataset à clean
def custom_standarization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


# On vectorise les données avec TextVectorization
# On définit aussi des constantes pour le modèle, comme sequence_length qui va permettre
# de tronquer les séquences en exactement 'sequence_length' valeurs
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standarization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Ensuite, on appelle 'adapt' pour adapter l'état du layer au dataset
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# On crée une fonction pour voir le résultat de l'utilisation du layer pour preprocess des données
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# On print un apercu (de 32 avis et labels) depuis le dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

# Comme on peut le voir ci-dessous, chaque token est remplacé par un entier
''' Output obtenu :
Review tf.Tensor(b'Great movie - especially the music - Etta James - "At Last". This speaks volumes when you have finally found that special someone.', shape=(), dtype=string)
Label neg
Vectorized review (<tf.Tensor: shape=(1, 250), dtype=int64, numpy=
array([[  86,   17,  260,    2,  222,    1,  571,   31,  229,   11, 2418,
           1,   51,   22,   25,  404,  251,   12,  306,  282,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]])>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
'''

# On peut voir le token (string) auquel chaque entier correspond en appelant .get_vocabulary()
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# On est prêts à entrainer le modèle
# On applique le layer de TextVectorization créé avant aux datasets d'entrainement, de validation et de test
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# On configure le dataset pour la performance
# .cache() garde les données en mémoire quand c'est supprimé du disque
# Si les données sont trop grandes, les données sont enregistrées sur le disque

# .prefetch() fait en sorte d'executer le preprocessing et le modele en meme temps lors de l'entainement
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

## CREATION DU MODELE ##
embedding_dim = 16

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])
model.summary()
'''
Explications : 
    - Le premier layer est un layer 'Embedding'.
    Ce layer prend les avis encodés en entiers (cf ligne 133) et regarde un vecteur pour chaque mot
    - Ensuite, le layer 'GlobalAveragePooling1D' retourne un vecteur de sortie ayant une taille de sortie fixée
      Cela permet au modèle d'avoir comme input une longueur de variable, de la manière la plus simple
'''

## Le modèle a besoin d'une fonction 'loss' et d'un optimisateur pour l'entrainement.
# Comme c'est une classification binaire et que le modele sort une probabilité,
# on utilise 'losses.BinaryCrossentropy' comme fonction de perte
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# on entraine le modele
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# on évalue le modèle
loss, accuracy = model.evaluate(test_ds)
print("Loss:", loss)
print("Accuracy:", accuracy)

## On créé un plot de l'accuracy et des pertes en fonction du temps
history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# Pertes
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

loss, accuracy = export_model.evaluate(raw_test_ds)
print(f"Accuracy avec le modele exporté : {accuracy}")

## On teste avec des nouvelles données
examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
]

print(f"Données prédites avec nouvelles données : {export_model.predict(examples)}")