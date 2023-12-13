import whisper
import gradio as gr
import getpass
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import VitsModel, AutoTokenizer
import torch
from scipy.io.wavfile import write
import numpy as np
import os

def transcribe(audio_file):
    model = whisper.load_model('base')
    result = model.transcribe(audio_file)
    return result['text']

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

### Load in CSV data containing transcripts and audio file names ###
loader = CSVLoader(file_path='chunk_text_db.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
})
data = loader.load()

### Functions to retrieve best matched audio chunk ###
def retrieve_audio_location(doc):
    split_doc = doc.rsplit(' ', 1)
    file_path = os.path.join(os.getcwd(), f"audio_chunks/{split_doc[1]}")
    return file_path

def retrieve_audio_chunk(openai_api_key, query):
    search = transcribe(query)
    vectorstore = Chroma.from_documents(documents=data, 
                                    embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(
    search
    )
    best_fit = retrieved_docs[0].page_content
    audio_path = retrieve_audio_location(doc=best_fit)
    return audio_path

### RAG Question-Answer system ###
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

def generate_rag_answer(openai_api_key, question):
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    
    vectorstore = Chroma.from_documents(documents=data, 
                                    embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

    retriever = vectorstore.as_retriever()
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(question)
    inputs = tokenizer(answer, return_tensors="pt")
    return inputs

def reformat_tensor(audio):
    audio_data = audio.reshape(-1).numpy()
    scaled_audio_data = audio_data * 32767
    scaled_audio_data = scaled_audio_data.astype('int16')
    return scaled_audio_data

def generate_qa_audio(openai_api_key, query):
    question = transcribe(query)
    inputs = generate_rag_answer(openai_api_key, question=question)

    with torch.no_grad():
        output = model(**inputs).waveform
        
    audio_data = reformat_tensor(output)

    write('answer.wav', 16000, audio_data) 
    
    file_path = os.path.join(os.getcwd(), 'answer.wav')
    return file_path

### Gradio app ###
search = gr.Interface(fn=retrieve_audio_chunk, 
            inputs=['text', gr.Audio(sources=['microphone', 'upload'], type='filepath')], 
            outputs='audio',
            description="Find audio clips about a specific topic.")

qa = gr.Interface(fn=generate_qa_audio, 
            inputs=['text', gr.Audio(sources=['microphone', 'upload'], type='filepath')], 
            outputs='audio',
            description="Ask a question and receive an answer from the biblical data.")

demo = gr.TabbedInterface([search, qa], ["Search", "QA"])
demo.launch()
