# Import required libraries
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone, Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
import os
import pinecone
import streamlit as st
import shutil
import json

OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
langchain.verbose = False


@st.cache_data()
def init():
    pinecone_index_name = ''
    chroma_collection_name = ''
    persist_directory = ''
    docsearch_ready = False
    directory_name = 'tmp_docs'
    return pinecone_index_name, chroma_collection_name, persist_directory, docsearch_ready, directory_name


@st.cache_data()
def save_file(files):
    # Remove existing files in the directory
    if os.path.exists(directory_name):
        for filename in os.listdir(directory_name):
            file_path = os.path.join(directory_name, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error: {e}")
    # Save the new file with original filename
    if files is not None:
        for file in files:
            file_name = file.name
            file_path = os.path.join(directory_name, file_name)
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file, f)


def load_files():
    all_texts = []
    n_files = 0
    n_char = 0
    n_texts = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    for filename in os.listdir(directory_name):
        file = os.path.join(directory_name, filename)
        if os.path.isfile(file):
            if file.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file)
            elif file.endswith(".pdf"):
                loader = PyMuPDFLoader(file)
            else:   # assume a pure text format and attempt to load it
                loader = UnstructuredFileLoader(file)
            data = loader.load()
            texts = text_splitter.split_documents(data)
            n_files += 1
            n_char += len(data[0].page_content)
            n_texts += len(texts)
            all_texts.extend(texts)
    st.write(
        f"Loaded {n_files} file(s) with {n_char} characters, and split into {n_texts} split-documents."
    )
    return all_texts, n_texts


@st.cache_resource()
def ingest(_all_texts, use_pinecone, _embeddings, pinecone_index_name, chroma_collection_name, persist_directory):
    if use_pinecone:
        docsearch = Pinecone.from_texts(
            [t.page_content for t in _all_texts], _embeddings, index_name=pinecone_index_name)  # add namespace=pinecone_namespace if provided
    else:
        docsearch = Chroma.from_documents(
            _all_texts, _embeddings, collection_name=chroma_collection_name, persist_directory=persist_directory)
    return docsearch


def setup_retriever(docsearch, k):
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": k}, include_metadata=True)
    return retriever


def setup_docsearch(use_pinecone, pinecone_index_name, embeddings, chroma_collection_name, persist_directory):
    docsearch = []
    n_texts = 0
    if use_pinecone:
        # Load the pre-created Pinecone index.
        # The index which has already be stored in pinecone.io as long-term memory
        if pinecone_index_name in pinecone.list_indexes():
            docsearch = Pinecone.from_existing_index(
                pinecone_index_name, embeddings)  # add namespace=pinecone_namespace if provided
            index_client = pinecone.Index(pinecone_index_name)
            # Get the index information
            index_info = index_client.describe_index_stats()
            namespace_name = ''
            n_texts = index_info['namespaces'][namespace_name]['vector_count']
        else:
            raise ValueError('''Cannot find the specified Pinecone index.
            				Create one in pinecone.io or using, e.g.,
            				pinecone.create_index(
            					name=index_name, dimension=1536, metric="cosine", shards=1)''')
    else:
        docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                           collection_name=chroma_collection_name)

        n_texts = docsearch._collection.count()

    return docsearch, n_texts


def get_response(query, chat_history, CRqa):
    result = CRqa({"question": query, "chat_history": chat_history})
    return result['answer'], result['source_documents']


def setup_em_llm(OPENAI_API_KEY, temperature, r_llm):
    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Use Open AI LLM with gpt-3.5-turbo or gpt-4.
    # Set the temperature to be 0 if you do not want it to make up things
    llm = ChatOpenAI(temperature=temperature, model_name=r_llm, streaming=True,
                     openai_api_key=OPENAI_API_KEY)
    return embeddings, llm


def load_chat_history(CHAT_HISTORY_FILENAME):
    try:
        with open(CHAT_HISTORY_FILENAME, 'r') as f:
            chat_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        chat_history = []
    return chat_history


def save_chat_history(chat_history, CHAT_HISTORY_FILENAME):
    with open(CHAT_HISTORY_FILENAME, 'w') as f:
        json.dump(chat_history, f)


pinecone_index_name, chroma_collection_name, persist_directory, docsearch_ready, directory_name = init()


def main(pinecone_index_name, chroma_collection_name, persist_directory, docsearch_ready, directory_name):
    docsearch_ready = False
    chat_history = []
    latest_chats = []
    reply = ''
    source = ''
    # Get user input of whether to use Pinecone or not
    col1, col2, col3 = st.columns([1, 1, 1])
    # create the radio buttons and text input fields
    with col1:
        r_pinecone = st.radio('Use Pinecone?', ('Yes', 'No'))
        r_ingest = st.radio(
            'Ingest file(s)?', ('Yes', 'No'))
        r_llm = st.multiselect(
            'LLM:', ['gpt-3.5-turbo', 'gpt-4'], 'gpt-3.5-turbo')
        r_llm = r_llm[0]
    with col2:
        OPENAI_API_KEY = st.text_input(
            "OpenAI API key:", type="password")
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1)
        k_sources = st.slider('# source(s) to print out', 0, 20, 2)
    with col3:
        if OPENAI_API_KEY:
            embeddings, llm = setup_em_llm(OPENAI_API_KEY, temperature, r_llm)
            if r_pinecone.lower() == 'yes':
                use_pinecone = True
                PINECONE_API_KEY = st.text_input(
                    "Pinecone API key:", type="password")
                PINECONE_API_ENV = st.text_input(
                    "Pinecone API env:", type="password")
                pinecone_index_name = st.text_input('Pinecone index:')
                pinecone.init(api_key=PINECONE_API_KEY,
                              environment=PINECONE_API_ENV)
            else:
                use_pinecone = False
                chroma_collection_name = st.text_input(
                    '''Chroma collection name of 3-63 characters:''')
                persist_directory = "./vectorstore"

    if pinecone_index_name or chroma_collection_name:
        session_name = pinecone_index_name + chroma_collection_name
        if r_ingest.lower() == 'yes':
            files = st.file_uploader(
                'Upload Files', accept_multiple_files=True)
            if files:
                save_file(files)
                all_texts, n_texts = load_files()
                docsearch = ingest(all_texts, use_pinecone, embeddings, pinecone_index_name,
                                   chroma_collection_name, persist_directory)
                docsearch_ready = True
        else:
            st.write(
                'No data is to be ingested. Make sure the Pinecone index or Chroma collection name you provided contains data.')
            docsearch, n_texts = setup_docsearch(use_pinecone, pinecone_index_name,
                                                 embeddings, chroma_collection_name, persist_directory)
            docsearch_ready = True
    if docsearch_ready:
        # number of sources (split-documents when ingesting files); default is 4
        k = min([20, n_texts])
        retriever = setup_retriever(docsearch, k)
        # CRqa = ConversationalRetrievalChain.from_llm(
        #     llm, retriever=retriever, return_source_documents=True)
        CRqa = ConversationalRetrievalChain(
            retriever=retriever, return_source_documents=True,
            question_generator=LLMChain(
                llm=llm, prompt=CONDENSE_QUESTION_PROMPT),
            combine_docs_chain=load_qa_chain(llm, chain_type="map_reduce"),
        )
        st.title(':blue[Chatbot]')
        # Get user input
        query = st.text_area('Enter your question:', height=10,
                             placeholder='''Summarize the context. 
                                            \nAfter typing your question, click on SUBMIT to send it to the bot.''')       
        submitted = st.button('SUBMIT')

        CHAT_HISTORY_FILENAME = f"chat_history/{session_name}_chat_hist.json"
        chat_history = load_chat_history(CHAT_HISTORY_FILENAME)
        st.markdown('<style>.my_title { font-weight: bold; color: red; }</style>', unsafe_allow_html=True)

        if query and submitted:
            # Generate a reply based on the user input and chat history
            chat_history = [(user, bot)
                            for user, bot in chat_history]
            reply, source = get_response(query, chat_history, CRqa)
            # Update the chat history with the user input and system response
            chat_history.append(('User', query))
            chat_history.append(('Bot', reply))
            save_chat_history(chat_history, CHAT_HISTORY_FILENAME)
            c = chat_history[-4:]
            if len(chat_history) >= 4:
                latest_chats = [c[2],c[3],c[0],c[1]]
            else:
                latest_chats = c

        if latest_chats:   
            chat_history_str1 = '<br>'.join([f'<span class=\"my_title\">{x[0]}:</span> {x[1]}' for x in latest_chats])        
            st.markdown(f'<div class=\"chat-record\">{chat_history_str1}</div>', unsafe_allow_html=True)

        if reply and source:
            # Display sources
            for i, source_i in enumerate(source):
                if i < k_sources:
                    if len(source_i.page_content) > 400:
                        page_content = source_i.page_content[:400]
                    else:
                        page_content = source_i.page_content
                    if source_i.metadata:
                        metadata_source = source_i.metadata['source']
                        st.markdown(f"<h3 class='my_title'>Source {i+1}: {metadata_source}</h3> <br> {page_content}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 class='my_title'>Source {i+1}: </h3> <br> {page_content}", unsafe_allow_html=True)

        all_chats = chat_history
        all_chat_history_str = '\n'.join(
                [f'{x[0]}: {x[1]}' for x in all_chats])
        st.title(':blue[All chat records]')
        st.text_area('', value=all_chat_history_str, height=250, label_visibility='collapsed')      
if __name__ == '__main__':
    main(pinecone_index_name, chroma_collection_name, persist_directory,
         docsearch_ready, directory_name)
