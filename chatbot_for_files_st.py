# Import required libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone, Chroma
from langchain.chains import ConversationalRetrievalChain
import os
import langchain
import pinecone
import streamlit as st
import shutil


OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
pinecone_index_name = ''
chroma_collection_name = ''
persist_directory = ''
docsearch_ready = False
directory_name = 'tmp_docs'
langchain.verbose = False


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


@st.cache_data()
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
        n_texts = docsearch._client._count(
            collection_name=chroma_collection_name)
    return docsearch, n_texts


def get_response(query, chat_history):
    result = CRqa({"question": query, "chat_history": chat_history})
    return result['answer'], result['source_documents']


def setup_em_llm(OPENAI_API_KEY, temperature):
    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Use Open AI LLM with gpt-3.5-turbo.
    # Set the temperature to be 0 if you do not want it to make up things
    llm = ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo", streaming=True,
                     openai_api_key=OPENAI_API_KEY)
    return embeddings, llm


# Get user input of whether to use Pinecone or not
col1, col2, col3 = st.columns([1, 1, 1])
# create the radio buttons and text input fields
with col1:
    r_pinecone = st.radio('Use Pinecone?', ('Yes', 'No'))
    r_ingest = st.radio(
        'Ingest file(s)?', ('Yes', 'No'))
with col2:
    OPENAI_API_KEY = st.text_input(
        "OpenAI API key:", type="password")
    temperature = st.slider('Temperature', 0.0, 1.0, 0.1)
    k_sources = st.slider('# source(s) to print out', 0, 20, 2)
with col3:
    if OPENAI_API_KEY:
        embeddings, llm = setup_em_llm(OPENAI_API_KEY, temperature)
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
    chat_history = []
    if r_ingest.lower() == 'yes':
        files = st.file_uploader('Upload Files', accept_multiple_files=True)
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
    CRqa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, return_source_documents=True)

    st.title('Chatbot')
    # Get user input
    query = st.text_area('Enter your question:', height=10,
                         placeholder='Summarize the context.')
    if query:
        # Generate a reply based on the user input and chat history
        reply, source = get_response(query, chat_history)
        # Update the chat history with the user input and system response
        chat_history.append(('User', query))
        chat_history.append(('Bot', reply))
        chat_history_str = '\n'.join(
            [f'{x[0]}: {x[1]}' for x in chat_history])
        st.text_area('Chat record:', value=chat_history_str, height=250)
        # Display sources
        for i, source_i in enumerate(source):
            if i < k_sources:
                if len(source_i.page_content) > 400:
                    page_content = source_i.page_content[:400]
                else:
                    page_content = source_i.page_content
                if source_i.metadata:
                    metadata_source = source_i.metadata['source']
                    st.write(
                        f"**_Source {i+1}:_** {metadata_source}: {page_content}")
                    st.write(source_i.metadata)
                else:
                    st.write(f"**_Source {i+1}:_** {page_content}")

