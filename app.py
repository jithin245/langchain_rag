import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

def load_document():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.pdf")
    document = loader.load()    
    return document   

def split_document(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    return chunks

def embeddings_on_vectordb(chunks):
    vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
                collection_name="local-rag"
            )
    return vector_db

def create_chain(vector_db):
    llm = ChatOllama(model="mistral")
    QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your 
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions seperated by newlines.
            Original questions: {question}""",
    )
    retriever = MultiQueryRetriever.from_llm(
                            vector_db.as_retriever(),
                            llm,
                            prompt=QUERY_PROMPT
                )

    #RAG prompt
    template = """Answer the question based ONLY on the following context:
                    {context}
                    Question: {question}
                    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm 
                | StrOutputParser()
            )

    return chain
    

def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    try:        
        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc.read())
            
            document = load_document()
            
            for _file in TMP_DIR.iterdir():
                temp_file = TMP_DIR.joinpath(_file)
                temp_file.unlink()

            chunks = split_document(document)
            vector_db = embeddings_on_vectordb(chunks)
            st.session_state.chain = create_chain(vector_db)

    except Exception as e:
        st.error(f"An error occurred: {e}") 


def input_fields():
    st.session_state.source_docs = st.file_uploader(label="Upload documents", type="pdf", accept_multiple_files=True)

def boot():
    input_fields()
    
    st.button("Submit Documents", on_click=process_documents)

    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = st.session_state.chain.invoke(query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    boot()