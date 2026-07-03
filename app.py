import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template

# ---- make sure required NLTK data is available ----
_required_nltk = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
for pkg, path in _required_nltk.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

lemmatizer = WordNetLemmatizer()

# ---- load everything train.py saved ----
intents = json.load(open("intents.json"))
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

app = Flask(__name__)


def bag_of_words(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    bag = [1 if w in tokens else 0 for w in words]
    return np.array(bag)


def predict_intent(sentence, threshold=0.6):
    bow = bag_of_words(sentence)
    prediction = model.predict(np.array([bow]), verbose=0)[0]

    best_index = np.argmax(prediction)
    confidence = prediction[best_index]

    if confidence < threshold:
        return None  # not confident enough - treat as "I don't understand"

    return classes[best_index]


def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = str(data.get("message", "")).strip()

    if not user_msg:
        return jsonify({"reply": "Please enter a message."}), 400

    tag = predict_intent(user_msg)

    if tag is None:
        reply = "Sorry, I'm not sure I understand. Could you rephrase that?"
    else:
        reply = get_response(tag)

    return jsonify({"reply": reply, "intent": tag})


if __name__ == "__main__":
    app.run(debug=True, port=5000)