import json
import pickle
import nltk
import numpy as np
import uvicorn
from fastapi import (FastAPI, WebSocket, WebSocketDisconnect)
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
import tensorflow.keras.models as tf_models

nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

app = FastAPI()

lemmatizer = WordNetLemmatizer()

with open("../astra/dataset.json") as file:
    intents = json.load(file)

with open("../astra/words.pkl", "rb") as file:
    words = pickle.load(file)

with open("../astra/classes.pkl", "rb") as file:
    classes = pickle.load(file)

chatbot_model = tf_models.load_model("../astra/astra.h5")

def clean_up_sentence(sentence):
    # Splits the sentence into individual words
    sentence_words = nltk.word_tokenize(sentence)
    # Converts each word into it's base form (lemma) while making it lowercase
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # Tokenize and lemmatize the sentence
    sentence_words = clean_up_sentence(sentence)
    # Initializes a list of zeros with the same length as the vocabulary (words)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            # Checks if each word from the vocabulary (words) appears in the sentence
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Transforms the sentence into a bag of words
    bow = bag_of_words(sentence)
    # Feeds the bag of words to the trained model and gets the prediction probabilities for each intent
    res = chatbot_model.predict(np.array([bow]))[0]
    # Filters out predictions with a probability below 0.25 (ignoring less confident predictions)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sorts the predictions in descending order based on their probabilities
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents):
    # Extracts the intent with the highest probability from the prediction
    tag = intents_list[0]["intent"]
    for intent in intents["intents"]:
        # Searches the intents for the matching intent
        if intent["tag"] == tag:
            return {
                "meaning": intent["meaning"],
                "procedures": intent["procedures"],
                "relations": intent["relations"],
                "references": intent["references"],
            }

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: dict

@app.get('/')
async def health_check():
    return "This is Astra AI."

@app.post("/api/talk", response_model=ChatResponse)
async def talk(request: ChatRequest):
    ints = predict_class(request.message)
    response = get_response(ints, intents)
    return ChatResponse(response=response)

@app.websocket("/api/ws/talk")
async def talk_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            ints = predict_class(data)
            response = get_response(ints, intents)
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
