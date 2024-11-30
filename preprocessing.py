import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import string

# Descargar recursos necesarios de NLTK
nltk.download('punkt')

class TextPreprocessor:
    def __init__(self, language='spanish'):
        self.language = language
        self.stemmer = SnowballStemmer(language)
        self.english_stemmer = nltk.PorterStemmer()  # Add English stemmer
        self.label_encoder = LabelEncoder()
        self.threshold = 0.3  # Umbral de confianza para respuestas
        
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.intents = json.load(file)
            
    def clean_text(self, text):
        # Convertir a minúsculas y eliminar puntuación
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        return text
            
    def preprocess(self):
        words = []
        tags = []
        xy = []
        
        # Recorrer todos los intents
        for intent in self.intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            
            for pattern in intent['patterns']:
                # Limpiar y tokenizar el texto
                cleaned_pattern = self.clean_text(pattern)
                tokens = word_tokenize(cleaned_pattern)
                
                # Apply stemming based on language
                if self.language == 'spanish':
                    tokens = [self.stemmer.stem(word) for word in tokens]
                else:
                    tokens = [self.english_stemmer.stem(word) for word in tokens]
                
                words.extend(tokens)
                xy.append((tokens, tag))
                
        # Eliminar duplicados y ordenar
        words = sorted(list(set(words)))
        tags = sorted(list(set(tags)))
        
        # Crear el bag of words
        X_train = []
        y_train = []
        
        for (pattern_tokens, tag) in xy:
            bag = []
            for word in words:
                bag.append(1 if word in pattern_tokens else 0)
            
            X_train.append(bag)
            y_train.append(tag)
        
        # Convertir a numpy arrays
        X_train = np.array(X_train)
        
        # Codificar las etiquetas
        self.label_encoder.fit(tags)
        y_train = self.label_encoder.transform(y_train)
        y_train = np.array(y_train)
        
        return X_train, y_train, words, tags
        
    def text_to_bag(self, text):
        # Limpiar y tokenizar el texto de entrada
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Apply stemming based on language
        if self.language == 'spanish':
            tokens = [self.stemmer.stem(word) for word in tokens]
        else:
            tokens = [self.english_stemmer.stem(word) for word in tokens]
        
        # Crear bag of words
        bag = []
        for word in self.words:
            bag.append(1 if word in tokens else 0)
            
        return np.array([bag])
        
    def tag_to_response(self, tag, confidence):
        # Si la confianza es menor al umbral, usar respuesta desconocida
        if confidence < self.threshold:
            tag = "desconocido"
            
        # Obtener una respuesta aleatoria para el tag dado
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])
        
        # Si no se encuentra el tag, devolver respuesta desconocida
        for intent in self.intents['intents']:
            if intent['tag'] == "desconocido":
                return np.random.choice(intent['responses'])
