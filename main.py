from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import tkinter as tk
import numpy as np
import random
import pickle
import json
import nltk


# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open('intents.json').read())

# Load preprocessed data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
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
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I couldn't understand that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send_message(event=None):
    message = entry.get()
    chat_history.insert(tk.END, "| You: " + message + "\n")
    if message.lower() == "bye" or message.lower() == "goodbye":
        ints = predict_class(message)
        res = get_response(ints, intents)
        chat_history.insert(tk.END, "| Bot: " + res + "\n")
        chat_history.insert(tk.END, "The Program Ends here! JamesBot Off\n")
        root.quit()
    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        chat_history.insert(tk.END, "| JamesBot: " + res + "\n")
    entry.delete(0, tk.END)

# Create the main window
root = tk.Tk()
root.title("JamesBot")
root.configure(bg="black")

# Create a text area to display chat history
chat_history = tk.Text(root, wrap=tk.WORD, width=50, height=20, bg="black", fg="white")
chat_history.pack(expand=True, fill='both')

# Create an entry field for user input
entry = tk.Entry(root, width=50, bg="black", fg="white")
entry.pack(pady=10)

# Bind the entry field to the 'Return' key to send messages
entry.bind("<Return>", send_message)

# Create a button to send user input to the chatbot
send_button = tk.Button(root, text="Send", command=send_message, bg="black", fg="white")
send_button.pack()

# Start the main event loop
root.mainloop()
