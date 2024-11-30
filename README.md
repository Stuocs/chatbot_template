# Plantilla para Chatbot

Este repositorio contiene una plantilla básica para un chatbot con varias intenciones predefinidas. El chatbot está diseñado para responder a preguntas comunes, hacer ciertos cálculos y mantener conversaciones básicas.

## Contenido
- **intents.json**: Archivo JSON con las intenciones del chatbot y las posibles respuestas.
- **README.md**: Documento de introducción y guía para el uso del repositorio.

## Cómo usar
1. **Clona el repositorio**:
### En Linux
   ```bash
   git clone https://github.com/tu-usuario/Plantilla-Chatbot.git
   cd Plantilla-Chatbot
   python3 -m venv venv
   source ./venv/bin/activate
   pip3 install -r requirements.txt
   python3 chatbot.py
   ```
### En Windows
   ```bash
   git clone https://github.com/tu-usuario/Plantilla-Chatbot.git
   cd Plantilla-Chatbot
   pip3 install -r requirements.txt
   python3 chatbot.py
   ```
En Windows da error algunas veces al utilizar `pip3 install` por lo que puede que necesites utilizar: 
```bash
pip install -r requiremnts.txt
python chatbot.py
```
Esto se debe a que en muchas ocasiones windows instala por defecto la versión 2 de python en vez de la 3.

2. **Modifica training_data.json
