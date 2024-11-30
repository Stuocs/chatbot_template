# Chatbot Template

This repository contains a basic template for a chatbot with several predefined intents. The chatbot is designed to answer common questions, perform certain calculations, and maintain basic conversations.

## Content
- **data/training_data.json**: JSON file with the chatbot's intents and possible responses.
- **README.md**: Introduction and guide to using the repository.

## How to use
1. **Clone the repository**:
### On Linux
   ```bash
   git clone https://github.com/your-username/Chatbot-Template.git
   cd Chatbot-Template
   python3 -m venv venv
   source ./venv/bin/activate
   pip3 install -r requirements.txt
   python3 chatbot.py
   ```
### On Windows
   ```bash
   git clone https://github.com/your-username/Chatbot-Template.git
   cd Chatbot-Template
   pip3 install -r requirements.txt
   python3 chatbot.py
   ```
On Windows, you might encounter an error using `pip3 install`, so you may need to use:
```bash
pip install -r requirements.txt
python chatbot.py
```
This is because Windows often installs Python 2 by default instead of Python 3.

2. **Modify training_data.json**:
Inside `/data/training_data.json`, you will find the training data and predefined responses. The bot is designed to store conversations within `learning_data` and learn from them, improving its response accuracy and interaction style.

To modify its interaction manner, you should adjust `/data/training_data.json`.

## Credits
This chatbot template was created by [Alesber](https://github.com/Stuocs).

## License
This project is licensed under the MIT License. For more information, see the [MIT_LICENSE](LICENSE) file.

# Plantilla para Chatbot

Este repositorio contiene una plantilla básica para un chatbot con varias intenciones predefinidas. El chatbot está diseñado para responder a preguntas comunes, hacer ciertos cálculos y mantener conversaciones básicas.

## Contenido
- **data/training_data.json**: Archivo JSON con las intenciones del chatbot y las posibles respuestas.
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
pip install -r requirements.txt
python chatbot.py
```
Esto se debe a que en muchas ocasiones windows instala por defecto la versión 2 de python en vez de la 3.

2. **Modifica training_data.json**:
En el interior de `/data/training_data.json` se almacena su entrenamiento y las respuestas que tiene para dar predefinidas, el bot además está creado de tal manera que dentro de `learning_data` almacene las conversaciones y aprenda en base a eso, mejorando así su precisión en las respuestas y la manera en la que lo hace.

Para modificar su manera de interaccionar se debe modificar `/data/training_data.json`.
