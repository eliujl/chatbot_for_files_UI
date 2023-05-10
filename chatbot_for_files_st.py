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
import pinecone
import streamlit as st

# Set up OpenAI API key (from .bashrc, Windows environment variables, .env)
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Set up Pinecone env
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_API_ENV = os.environ['PINECONE_API_ENV']
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)


def load_files():
    file_path = "./docs/"
    all_texts = []
    n_files = 0
    n_char = 0
    n_texts = 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    for filename in os.listdir(file_path):
        file = os.path.join(file_path, filename)
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
    print(
        f"Loaded {n_files} file(s) with {n_char} characters, and split into {n_texts} split-documents."
    )
    return all_texts, n_texts


def ingest(all_texts, use_pinecone, embeddings, pinecone_index_name, chroma_collection_name, persist_directory):
    if use_pinecone:
        docsearch = Pinecone.from_texts(
            [t.page_content for t in all_texts], embeddings, index_name=pinecone_index_name)  # add namespace=pinecone_namespace if provided
    else:
        docsearch = Chroma.from_documents(
            all_texts, embeddings, collection_name=chroma_collection_name, persist_directory=persist_directory)
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
            pass
            # raise ValueError('''Cannot find the specified Pinecone index.
            # 				Create one in pinecone.io or using
            # 				pinecone.create_index(
            # 					name=index_name, dimension=1536, metric="cosine", shards=1)''')
    else:
        docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                           collection_name=chroma_collection_name)
        n_texts = docsearch._client._count(
            collection_name=chroma_collection_name)
    return docsearch, n_texts


def get_response(query, chat_history):
    result = CRqa({"question": query, "chat_history": chat_history})
    return result['answer'], result['source_documents']


def setup_em_llm(OPENAI_API_KEY):

    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Use Open AI LLM with gpt-3.5-turbo.
    # Set the temperature to be 0 if you do not want it to make up things
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True,
                     openai_api_key=OPENAI_API_KEY)
    return embeddings, llm


pinecone_index_name = ''
chroma_collection_name = ''
persist_directory = ''
chat_history = []


# Get user input of whether to use Pinecone or not
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
# create the radio buttons and text input fields
with col1:
    r_pinecone = st.radio('Do you want to use Pinecone index?', ('Yes', 'No'))
with col2:
    r_ingest = st.radio(
        'Do you want to ingest the file(s) in ./docs/?', ('Yes', 'No'))
with col3:
    OPENAI_API_KEY = st.text_input(
        "Enter your OpenAI API key and press Enter", type="password")
with col4:
    if OPENAI_API_KEY:
        embeddings, llm = setup_em_llm(OPENAI_API_KEY)
        if r_pinecone.lower() == 'yes' and PINECONE_API_KEY != '':
            use_pinecone = True
            pinecone_index_name = st.text_input('Enter your Pinecone index')
        else:
            use_pinecone = False
            chroma_collection_name = st.text_input(
                'Not using Pinecone or empty Pinecone API key provided. Using Chroma. Enter Chroma collection name of 3-63 characters:')
            persist_directory = "./vectorstore"

if pinecone_index_name or chroma_collection_name:
    if r_ingest.lower() == 'y':
        all_texts, n_texts = load_files()
        docsearch = ingest(all_texts, use_pinecone, embeddings, pinecone_index_name,
                           chroma_collection_name, persist_directory)
    else:
        st.write(
            'No data is to be ingested. Make sure the Pinecone index or Chroma collection name you provided contains data.')
        docsearch, n_texts = setup_docsearch(use_pinecone, pinecone_index_name,
                                             embeddings, chroma_collection_name, persist_directory)
    # number of sources (split-documents when ingesting files); default is 4
    k = min([20, n_texts])
    retriever = setup_retriever(docsearch, k)
    CRqa = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, return_source_documents=True)

    st.title('Chatbot')

    # Get user input
    query = st.text_input('Enter your question; enter "exit" to exit')

    if query:
        # Generate a reply based on the user input and chat history
        reply, source = get_response(query, chat_history)

        # Update the chat history with the user input and system response
        chat_history.append(('User: ', query))
        chat_history.append(('Bot: ', reply))
        chat_history_str = '\n\n'.join(
            [f'{x[0]}: {x[1]}' for x in chat_history])

        st.text_area('Chat record:', value=chat_history_str, height=250)

        # Display sources
        for i, source_i in enumerate(source):
            if i < 2:
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
