import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# Download 'punkt' tokenizer
nltk.download('punkt')

# Define sports intents
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello!", "Hi there!", "Hey!"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye"],
            "responses": ["Goodbye!", "See you later!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "Thanks a lot"],
            "responses": ["You're welcome!", "No problem!"]
        },
        {
            "tag": "sports_news",
            "patterns": ["What's the latest sports news?", "Tell me sports news", "Update me on sports"],
            "responses": ["Here is the latest sports news..."]
        },
        {
            "tag": "sports_scores",
            "patterns": ["What's the score of the game?", "Tell me the latest scores", "What's the score?"],
            "responses": ["Here are the latest scores..."]
        },
        {
            "tag": "player_stats",
            "patterns": ["Tell me about player stats", "Player statistics", "What are the stats for player X?"],
            "responses": ["Here are the stats for the player..."]
        }
    ]
}

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# save words and classes
with open('sports_words.pkl', 'wb') as file:
    pickle.dump(words, file)

with open('sports_classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

print("Words:", words)
print("Classes:", classes)
