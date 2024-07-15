# Importaciones y librerías necesarias

import os
import torch
import logging
import chainlit as cl
from transformers import AutoTokenizer, ConversationalPipeline, Pipeline
from typing import List
from getpass import getpass
from chainlit.types import AskFileResponse
from langchain_community.chat_models import ChatHuggingFace
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_huggingface.chat_models import ChatHuggingFace
from huggingface_hub import HfApi, InferenceApi, login
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


# Configuración del token de Hugging Face e inicio de sesión
HUGGINGFACE_API_TOKEN = "Api_Token"
login(token=HUGGINGFACE_API_TOKEN) 


# Se crea un objeto RecursiveCharacterTextSplitter con un tamaño de fragmento de 1000 y un solapamiento de 100
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# Se define la plantilla del sistema para las respuestas, esta plantilla se usa para crear SystemMessagePromptTemplate y HumanMessagePromptTemplate
# Y luego se combina en un ChatPromptTemplate
system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
And if the user greets with greetings like Hi, hello, How are you, etc reply accordingly as well.
Example of your response should be:
The answer is foo
SOURCES: xyz
Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


# Esta función recibe un archivo PDF y devuelve una lista de fragmentos de texto extraídos del archivo
def process_file(file: AskFileResponse):
    file_path = file.path
    pypdf_loader = PyPDFLoader(file_path)
    texts = pypdf_loader.load_and_split()
    texts = [text.page_content for text in texts]
    return texts


# Configuración del nivel de logging
logging.basicConfig(level=logging.DEBUG)


# Esta función asincrónica se encarga de autenticar y obtener un objeto HuggingFaceEndpoint, se autentica con el token de Hugging Face
async def get_huggingface_llm(repo_id: str, api_key: str):
    logging.debug(f"Creating HuggingFaceEndpoint with repo_id={repo_id} and token={HUGGINGFACE_API_TOKEN}")
    login(token=HUGGINGFACE_API_TOKEN, add_to_git_credential=True)
    llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HUGGINGFACE_API_TOKEN)
    logging.debug("Autenticación exitosa")
    return llm


# Esta función se ejecuta al inicio de la conversación
# Solicita al usuario que cargue un archivo PDF y procesa el archivo para extraer los fragmentos de texto
# Luego crea un objeto Chroma con los fragmentos de texto y un objeto ChatHuggingFace con un modelo de lenguaje de Hugging Face
# Finalmente, crea una cadena de conversación con recuperación que utiliza el modelo de lenguaje y el objeto Chroma
# Informa al usuario que el sistema está listo para responder preguntas
@cl.on_chat_start
async def on_chat_start():
    files = None

    # Esperar a que el usuario cargue un archivo
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, carga un archivo PDF para comenzar.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Procesando `{file.name}`...")
    await msg.send()

    # Cargar el archivo
    pypdf_loader = PyPDFLoader(file.path)
    texts = pypdf_loader.load_and_split()
    texts = [text.page_content for text in texts]

    # Crear metadatos para cada fragmento
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Crear embeddings de Hugging Face
    embeddings = HuggingFaceEmbeddings()

    # Crear una instancia asincrónica de Chroma y cargar los textos
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Crear LLM de Hugging Face
    HUGGINGFACE_API_TOKEN = "HGF_API_TOKEN"
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    logging.debug(f"Iniciando on_chat_start con repo_id={repo_id} y token: {HUGGINGFACE_API_TOKEN}")
    llm = await get_huggingface_llm(repo_id, HUGGINGFACE_API_TOKEN)

    # Crear una cadena de conversación con recuperación
    chain = ConversationalRetrievalChain.from_llm(
        ChatHuggingFace(llm=llm, temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Informar al usuario que el sistema está listo
    msg.content = f"Procesamiento de `{file.name}` completado. ¡Ahora puedes hacer preguntas!"
    await msg.update()

    cl.user_session.set("chain", chain)


# Esta función se ejecuta cuando el usuario envía un mensaje
# La función obtiene la cadena de conversación de la sesión del usuario
# Luego, utiliza la cadena de conversación para responder al mensaje del usuario
# Envía la respuesta al usuario y, si es necesario, envía los fragmentos de texto referenciados en la respuesta
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  
    if chain is None:
        await cl.Message(content="The chain is not initialized properly.").send() # Si la cadena no está inicializada, se envía un mensaje de error
        return

    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  

    text_elements = []  

    #if source_documents:
    #    for source_idx, source_doc in enumerate(source_documents):
    #        source_name = f"source_{source_idx}"
            # Crear el elemento de texto referenciado en el mensaje
    #        text_elements.append(
    #            cl.Text(content=source_doc.page_content, name=source_name)
    #        )
    #    source_names = [text_el.name for text_el in text_elements]

        #if source_names:
        #    answer += f"\nSources: {', '.join(source_names)}"
        #else:
        #    answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

