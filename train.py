import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# ---- one-time downloads (safe to re-run) ----
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# ---- load training data ----
with open("intents.json") as f:
    intents = json.load(f)

words = []       # every unique word seen (vocabulary)
classes = []      # every intent tag (e.g. "greeting", "delivery")
documents = []    # (tokenized pattern, tag) pairs
ignore = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatize + clean + dedupe vocabulary
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"{len(documents)} patterns, {len(classes)} intents, {len(words)} unique words")

# ---- build training data: bag-of-words -> intent (one-hot) ----
training = []
output_empty = [0] * len(classes)

for tokens, tag in documents:
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    bag = [1 if w in pattern_words else 0 for w in words]

    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1

    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([row[0] for row in training])
train_y = np.array([row[1] for row in training])

# ---- simple ANN (this is the "TensorFlow ANN" part) ----
model = Sequential([
    Dense(64, input_shape=(len(train_x[0]),), activation="relu"),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(len(train_y[0]), activation="softmax")   # one output per intent
])

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# ---- save everything the API needs later ----
model.save("chatbot_model.h5")
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

print("Training done! Files saved: chatbot_model.h5, words.pkl, classes.pkl")
