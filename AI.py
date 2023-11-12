import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data from text files
with open('Train/Train_words.txt', 'r') as file:
    words = file.read().splitlines()
    print(words)

with open('Train/Train_spellings.txt', 'r') as file:
    spellings = [[i for i in line.split()] for line in file.read().splitlines()]
    print(spellings)

# Tokenize words
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(words)
words_sequences = word_tokenizer.texts_to_sequences(words)

max_word_length = max(len(word) for word in words)
padded_words = pad_sequences(words_sequences, maxlen=max_word_length, padding='post')

# Tokenize spellings
spelling_tokenizer = Tokenizer(char_level=True)
spelling_tokenizer.fit_on_texts(spellings)
spellings_sequences = spelling_tokenizer.texts_to_sequences(spellings)

max_spelling_length = max(len(spell) for spell in spellings)
padded_spellings = pad_sequences(spellings_sequences, maxlen=max_spelling_length, padding='post')

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=50, input_length=max_word_length))
model.add(LSTM(100))
model.add(Dense(max_spelling_length, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_words, padded_spellings, epochs=10, batch_size=32, validation_split=0.2)

# Save the model if needed
model.save('spellings_model.keras')