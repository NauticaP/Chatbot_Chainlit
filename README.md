# Chatbot con sistema QA

Este proyecto permite a los usuarios cargar un archivo PDF y hacer preguntas sobre su contenido utilizando un modelo de lenguaje de Hugging Face. El sistema responde utilizando fragmentos de texto relevantes del PDF como fuente y mantiene un historial de la conversación.

## Requisitos

- Python 3.7+
- pip

## Instalación

  1. Crea un entorno virtual
  ```bash
  python -m venv venv
  source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
  ```

  2. Instala las dependencias
  ```bash
  pip install -r requirements.txt
  ```

  3. Configura tu token de Hugging Face:  
  Abre el archivo chat4.2_app.py y reemplaza "HGF_API_TOKEN" con tu token de Hugging Face.

  5. Ejecuta el script:
  ```bash
  chainlit run chat4.2_app.py
  ```

## Estructura del código
  1. Importaciones y configuración:  
    - Importa librerías necesarias y configura el token de Hugging Face.

  2. Definición de funciones:  
    - process_file: Procesa un archivo PDF y lo divide en fragmentos de texto.  
    - get_huggingface_llm: Crea y retorna un modelo de lenguaje Hugging Face.

  3. Configuración del evento de inicio del chat (on_chat_start):  
    - Solicita un archivo PDF al usuario, lo procesa y crea un sistema de búsqueda basado en embeddings.

  4. Manejo de mensajes del usuario (on_message):  
    - Procesa preguntas del usuario utilizando la cadena de recuperación conversacional y responde con fragmentos relevantes del PDF.

## Ejemplo de uso
  1. Al abrirse tu navegador web y carga un archivo PDF cuando se te solicite.
  2. Haz preguntas sobre el contenido del PDF y recibe respuestas basadas en los fragmentos del documento.
