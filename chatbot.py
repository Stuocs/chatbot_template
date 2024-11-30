import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from preprocessing import TextPreprocessor
from model import ChatbotModel
import os
import re

class Chatbot:
    def __init__(self, model_path='model.pth', learning_data_path='learning_data.txt'):
        self.preprocessor = TextPreprocessor()
        self.model_path = model_path
        self.learning_data_path = learning_data_path
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    def train(self, data_path='data/training_data.json'):
        # Cargar y preprocesar datos
        self.preprocessor.load_data(data_path)
        X_train, y_train, words, tags = self.preprocessor.preprocess()
        
        # Guardar words y tags en el preprocessor para uso posterior
        self.preprocessor.words = words
        self.preprocessor.tags = tags
        
        # Crear y entrenar el modelo
        self.model = ChatbotModel(
            input_size=len(words),
            hidden_size=16,
            output_size=len(tags)
        )
        
        print("Iniciando entrenamiento...")
        self.model.train(X_train, y_train, epochs=1000)
        print("Entrenamiento completado!")
        
        # Guardar el modelo
        self.model.save_model(self.model_path)
        
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")
            
        # Cargar datos para el preprocessor
        self.preprocessor.load_data('data/training_data.json')
        _, _, words, tags = self.preprocessor.preprocess()
        self.preprocessor.words = words
        self.preprocessor.tags = tags
            
        self.model = ChatbotModel.load_model(self.model_path)
        
    def detect_language(self, text):
        # Simple language detection based on keywords
        spanish_keywords = ['hola', 'adiós', 'gracias', 'cuéntame']
        english_keywords = ['hello', 'bye', 'thank you', 'tell me']
        
        if any(keyword in text.lower() for keyword in spanish_keywords):
            return 'spanish'
        elif any(keyword in text.lower() for keyword in english_keywords):
            return 'english'
        return 'unknown'

    def get_response(self, text):
        language = self.detect_language(text)
        response = ""
        
        if language == 'spanish':
            # Manejar preguntas matemáticas
            if self.is_math_question(text):
                return self.solve_math_question(text)
            
            # Manejar solicitudes de cuentos
            if "cuéntame un cuento" in text.lower():
                return self.generate_story()
            
            # Convertir texto a bag of words
            X = self.preprocessor.text_to_bag(text)
            
            # Obtener predicción y confianza
            pred_idx, confidence = self.model.predict(X)
            
            # Convertir índice predicho a tag
            predicted_tag = self.preprocessor.label_encoder.inverse_transform([pred_idx])[0]
            
            # Obtener respuesta para el tag, considerando la confianza
            response = self.preprocessor.tag_to_response(predicted_tag, confidence)
            
            # Si la confianza es baja, usar el modelo generativo
            if confidence < 0.5:  # Umbral de confianza
                response = self.generate_response(text)
        
        elif language == 'english':
            # Handle English responses similarly
            # Convert text to bag of words
            X = self.preprocessor.text_to_bag(text)
            
            # Get prediction and confidence
            pred_idx, confidence = self.model.predict(X)
            
            # Convert predicted index to tag
            predicted_tag = self.preprocessor.label_encoder.inverse_transform([pred_idx])[0]
            
            # Get response for the tag, considering confidence
            response = self.preprocessor.tag_to_response(predicted_tag, confidence)
            
            # If confidence is low, use the generative model
            if confidence < 0.5:  # Confidence threshold
                response = self.generate_response(text)
        
        # Store interaction for future learning
        self.store_learning_data(text, response)
        
        return response
        # Manejar preguntas matemáticas
        if self.is_math_question(text):
            return self.solve_math_question(text)
        
        # Manejar solicitudes de cuentos
        if "cuéntame un cuento" in text.lower():
            return self.generate_story()
        
        # Convertir texto a bag of words
        X = self.preprocessor.text_to_bag(text)
        
        # Obtener predicción y confianza
        pred_idx, confidence = self.model.predict(X)
        
        # Convertir índice predicho a tag
        predicted_tag = self.preprocessor.label_encoder.inverse_transform([pred_idx])[0]
        
        # Obtener respuesta para el tag, considerando la confianza
        response = self.preprocessor.tag_to_response(predicted_tag, confidence)
        
        # Si la confianza es baja, usar el modelo generativo
        if confidence < 0.5:  # Umbral de confianza
            response = self.generate_response(text)
        
        # Almacenar la interacción para aprendizaje futuro
        self.store_learning_data(text, response)
        
        return response
    
    def is_math_question(self, text):
        # Detectar si la pregunta es matemática
        return bool(re.search(r'\b\d+\s*[+\-*/]\s*\d+\b', text))
    
    def solve_math_question(self, text):
        # Evaluar la expresión matemática
        try:
            result = eval(text)
            return f"El resultado de {text} es {result}."
        except Exception as e:
            return "Lo siento, no puedo resolver esa operación."
    
    def generate_response(self, prompt):
        # Generar respuesta usando GPT-2
        inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.gpt2_model.generate(inputs, max_length=50, num_return_sequences=1)
        response = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def generate_story(self):
        # Generar un cuento usando GPT-2
        prompt = "Érase una vez un valiente caballero que"
        inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.gpt2_model.generate(inputs, max_length=150, num_return_sequences=1)
        story = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return story
    
    def store_learning_data(self, user_input, bot_response):
        # Almacenar la interacción en un archivo para aprendizaje futuro
        with open(self.learning_data_path, 'a') as f:
            f.write(f"Usuario: {user_input}\nBot: {bot_response}\n\n")

def main():
    # Eliminar el modelo existente para reentrenar
    if os.path.exists('model.pth'):
        os.remove('model.pth')
        
    chatbot = Chatbot()
    
    # Verificar si existe un modelo guardado
    if not os.path.exists('model.pth'):
        print("Entrenando nuevo modelo...")
        chatbot.train()
    else:
        print("Cargando modelo existente...")
        chatbot.load()
    
    print("\n¡Chatbot listo!")
    print("Puedo responder a saludos, despedidas, agradecimientos y preguntas sobre mis capacidades.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        user_input = input("Tú: ")
        
        if user_input.lower() in ['salir', 'exit', 'quit', 'bye']:
            print("¡Hasta luego!")
            break
            
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
