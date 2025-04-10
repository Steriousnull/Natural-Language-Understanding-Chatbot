from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
import json
import nltk



lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [0] * len(words)  # Initialize bag with zeros
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        if word in word_patterns:
            bag[words.index(word)] = 1  # Update bag if word is present

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append((bag, output_row))  # Append a tuple of bag and output row

random.shuffle(training)

# Split the training data into input features and target labels
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Define the SGD optimizer without decay
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

# Compile your model using the defined optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(train_x, train_y, epochs=300, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('Done')
