import io
import json
import pickle
import random
from typing import List

import nltk
import numpy as np
import uvicorn
from fastapi import (FastAPI, File, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
import tensorflow.keras.models as tf_models

nltk.download('all')

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

app = FastAPI()

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open("./astra/dataset.json") as file:
    intents = json.load(file)

# Load words, classes, and model (using TensorFlow for the chatbot)
with open("./astra/words.pkl", "rb") as file:
    words = pickle.load(file)

with open("./astra/classes.pkl", "rb") as file:
    classes = pickle.load(file)

# Load the TensorFlow chatbot model explicitly
chatbot_model = tf_models.load_model("./astra/astra.h5")

# Text model functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = chatbot_model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
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

@app.post("/api/talk", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    ints = predict_class(request.message)
    response = get_response(ints, intents)
    return ChatResponse(response=response)

@app.websocket("/api/ws/talk")
async def websocket_chat(websocket: WebSocket):
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
