# @English_Version
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

# @Versión_En_Español
# Plantilla para Chatbot

Este repositorio contiene una plantilla básica para un chatbot con varias intenciones predefinidas. El chatbot está diseñado para responder a preguntas comunes, realizar ciertos cálculos y mantener conversaciones básicas.

## Contenido
- **data/training_data.json**: Archivo JSON con las intenciones del chatbot y las posibles respuestas.
- **README.md**: Introducción y guía para el uso del repositorio.

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
En Windows, podrías encontrarte con un error al usar `pip3 install`, por lo que podrías necesitar usar:
```bash
pip install -r requirements.txt
python chatbot.py
```
Esto se debe a que Windows a menudo instala Python 2 por defecto en lugar de Python 3.

2. **Modifica training_data.json**:
Dentro de `/data/training_data.json`, encontrarás los datos de entrenamiento y las respuestas predefinidas. El bot está diseñado para almacenar conversaciones dentro de `learning_data` y aprender de ellas, mejorando su precisión de respuesta y estilo de interacción.

Para modificar la manera en que interactúa, debes ajustar `/data/training_data.json`.

## Créditos
Esta plantilla de chatbot fue creada por [Alesber](https://github.com/Stuocs).

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Para más información, consulta el archivo [MIT_LICENSE](LICENSE).
