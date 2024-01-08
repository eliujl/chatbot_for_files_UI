# Import required libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone, Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import langchain
import pinecone
import streamlit as st
import shutil
import json

OPENAI_API_KEY = ''
PINECONE_API_KEY = ''
PINECONE_API_ENV = ''
gpt3p5 = 'gpt-3.5-turbo-1106'
gpt4 = 'gpt-4-1106-preview'
local_model_tuples = [
        (0, 'mistral_7b', "TheBloke/OpenHermes-2-Mistral-7B-GGUF", "openhermes-2-mistral-7b.Q8_0.gguf", "mistral", "https://huggingface.co/TheBloke/OpenHermes-2-Mistral-7B-GGUF"),
        (1, 'mistral_7b_inst_small', "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", "mistral-7b-instruct-v0.1.Q2_K.gguf", "mistral", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF"),
        (2, 'mistral_7b_inst_med', "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", "mistral-7b-instruct-v0.1.Q8_0.gguf", "mistral", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF"),
        (3, 'llama_13b_small', "TheBloke/Llama-2-13B-chat-GGUF", "llama-2-13b-chat.Q4_K_M.gguf", "llama", "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF"),
        (4, 'llama_13b_med', "TheBloke/Llama-2-13B-chat-GGUF", "llama-2-13b-chat.Q8_0.gguf", "llama", "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF"),
        (5, 'mixtral', "TheBloke/Mixtral-8x7B-v0.1-GGUF", "mixtral-8x7b-v0.1.Q8_0.gguf", "mixtral", "https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF"),
        (6, 'mixtral_inst', "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "mixtral-8x7b-instruct-v0.1.Q2_K.gguf", "mixtral", "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"),
    ]
local_model_names = [t[1] for t in local_model_tuples]
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
            # namespace_name = ''
            # if index_info is not None:
            #     print(index_info['namespaces'][namespace_name]['vector_count'])
            # else:
            #     print("Index information is not available.")            
            # n_texts = index_info['namespaces'][namespace_name]['vector_count']
            n_texts = index_info['total_vector_count']
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


@st.cache_resource()
def use_local_llm(r_llm, local_llm_path):
    from langchain.llms import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from huggingface_hub import hf_hub_download
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    entry = local_model_names.index(r_llm)
    model_id, local_model_name, model_name, model_file, model_type, model_link = local_model_tuples[entry]
    model_path = os.path.join( local_llm_path, model_name, model_file )
    model_path = os.path.normpath( model_path )
    model_dir = os.path.join( local_llm_path, model_name )
    model_dir = os.path.normpath( model_dir )
    if not os.path.exists(model_path):
        print("model not existing at ", model_path, "\n")
        model_path = hf_hub_download(repo_id=model_name, filename=model_file, repo_type="model",
                #cache_dir=local_llm_path, 
                #local_dir=local_llm_path, 
                local_dir=model_dir,
                local_dir_use_symlinks=False)
        print("\n model downloaded at path=",model_path)
    else:
        print("model existing at ", model_path)
    
    llm = LlamaCpp( 
        model_path=model_path,
        # temperature=0.0,
        # n_batch=300,
        n_ctx=4000,
        max_tokens=2000,
        # n_gpu_layers=10,
        # n_threads=12,
        # top_p=1,
        # repeat_penalty=1.15,
        # verbose=False,
        # callback_manager=callback_manager, 
        # streaming=True,
        # chat_format="llama-2",
        # verbose=True, # Verbose is required to pass to the callback manager
    )
    return llm


def setup_prompt(r_llm, usage):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS_LLAMA, E_SYS_LLAMA = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_SYS_MIS, E_SYS_MIS = "<s> ", "</s> "
    B_SYS_MIXTRAL, E_SYS_MIXTRAL = "<s>[INST]", "[/INST]</s>[INST]"
    system_prompt_rag = """Answer the question in your own words as truthfully as possible from the context given to you.
        Supply sufficient information, evidence, reasoning, source from the context, etc., to justify your answer with details and logic.
        Think step by step and do not jump to conclusion during your reasoning at the beginning.
        Sometimes user's question may appear to be directly related to the context but may still be indirectly related, 
            so try your best to understand the question based on the context and chat history.
        If questions are asked where there is no relevant context available, 
            respond using out-of-context knowledge with 
            "This question does not seem to be relevant to the documents. I am trying to explore knowledge outside the context." """
    system_prompt_chat = """Answer the question in your own words.
        Supply sufficient information, evidence, reasoning, source from the context, etc., to justify your answer with details and logic.
        Think step by step and do not jump to conclusion during your reasoning at the beginning.
        """
    system_prompt_task = """You will be given a task, and you are an expert in that task. 
        Perform the task for the given context, and output the result. Do not include extra descriptions. Just output the desired result defined by the task.
        Example: You are a professional translator and are given a translation task. Then you translate the text in the context and output only the translated text.
        Example: You are a professional proofreader and are given a proofreading task. Then you proofread the text in the context and output only the translated text.
        """    
    if usage == 'RAG':
        system_prompt = system_prompt_rag
        instruction = """
            Context: {context}

            Chat history: {chat_history}
            User: {question}
            Bot: answer """
    elif usage == 'Chat':
        system_prompt = system_prompt_chat
        instruction = """
            Chat history: {chat_history}
            User: {question}
            Bot: answer """
    elif usage == 'Task':
        system_prompt = system_prompt_task
        instruction = """
            Context: {context}
            User: {question}
            Bot: answer """        
    if r_llm == gpt3p5 or r_llm == gpt4:
        template = system_prompt + instruction
    else:
        entry = local_model_names.index(r_llm)
        if local_model_tuples[entry][4] == 'llama':
            template = B_INST + B_SYS_LLAMA + system_prompt + E_SYS_LLAMA + instruction + E_INST
        elif local_model_tuples[entry][4] == 'mistral':
            template = B_SYS_MIS + B_INST + system_prompt + E_INST + E_SYS_MIS + B_INST + instruction + E_INST
        elif local_model_tuples[entry][4] == 'mixtral':
            template = B_SYS_MIXTRAL + system_prompt + E_SYS_MIXTRAL + B_INST + instruction + E_INST
        else:
            # Handle other models or raise an exception
            pass
    if usage == 'RAG':
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"], template=template
        )
    elif usage == 'Chat':
        prompt = PromptTemplate(
            input_variables=["chat_history", "question"], template=template
        )
    elif usage == 'Task':
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=template
        )
    return prompt

def setup_em_llm(OPENAI_API_KEY, temperature, r_llm, local_llm_path):
    if (r_llm == gpt3p5 or r_llm == gpt4) and OPENAI_API_KEY:
        # Set up OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # Use Open AI LLM with gpt-3.5-turbo or gpt-4.
        # Set the temperature to be 0 if you do not want it to make up things
        llm = ChatOpenAI(temperature=temperature, model_name=r_llm, streaming=True,
                        openai_api_key=OPENAI_API_KEY)    
    else:     
        #em_model_name = 'hkunlp/instructor-xl'
        em_model_name='sentence-transformers/all-mpnet-base-v2'
        embeddings = HuggingFaceEmbeddings(model_name=em_model_name)
        llm = use_local_llm(r_llm, local_llm_path)
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
    LLMs = [gpt3p5, gpt4] + local_model_names
    usage = 'RAG'
    local_llm_path = './models/'
    user_llm_path = ''
    # Get user input of whether to use Pinecone or not
    col1, col2, col3 = st.columns([1, 1, 1])
    # create the radio buttons and text input fields
    with col1:
        usage = st.radio('Usage: RAG for ingested files, chat (no files), or task (for all ingested texts)', ('RAG', 'Chat', 'Task'))
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1)
        if usage == 'RAG':
            r_pinecone = st.radio('Vector store:', ('Pinecone (online)', 'Chroma (local)'))
            k_sources = st.slider('# source(s) to print out', 0, 20, 2)
            r_ingest = st.radio('Ingest file(s)?', ('Yes', 'No'))
            if r_pinecone == 'Pinecone (online)':
                use_pinecone = True
            else:
                use_pinecone = False                
        if usage == 'Task':
            r_ingest = 'Yes'
       
    with col2:
        r_llm = st.radio(label='LLM:', options=LLMs)
        if r_llm == gpt3p5 or r_llm == gpt4:
            use_openai = True
        else:
            use_openai = False             
        if use_openai == True:
            OPENAI_API_KEY = st.text_input(
                "OpenAI API key:", type="password")
        else:
            OPENAI_API_KEY = ''
            if usage == 'RAG' and use_pinecone == True:
                st.write('Local GPT model (and local embedding model) is selected. Online vector store is selected.')
            elif usage == 'RAG' and use_pinecone == False:
                st.write('Local GPT model (and local embedding model) and local vector store are selected. All info remains local.')
            else:
                st.write('Local GPT model is selected. All info remains local.')
    with col3:
        if usage == 'RAG':
            if use_pinecone == True:
                PINECONE_API_KEY = st.text_input(
                    "Pinecone API key:", type="password")
                PINECONE_API_ENV = st.text_input(
                    "Pinecone API env:", type="password")
                pinecone_index_name = st.text_input('Pinecone index:')
                pinecone.init(api_key=PINECONE_API_KEY,
                                environment=PINECONE_API_ENV)
            else:
                chroma_collection_name = st.text_input(
                    '''Chroma collection name of 3-63 characters:''')
                persist_directory = "./vectorstore"
        else:
            hist_fn = st.text_input('Chat history filename')
        if use_openai == False:
            user_llm_path = st.text_input(
                "Path for local model (TO BE DOWNLOADED IF NOT EXISTING), type 'default' to use default path:",
                placeholder="default")
            if 'default' in user_llm_path:
                user_llm_path = local_llm_path

    if ( (pinecone_index_name or chroma_collection_name or usage == 'Task' or usage == 'Chat') 
        and ( (use_openai and OPENAI_API_KEY) or (not use_openai and user_llm_path) ) ):
        embeddings, llm = setup_em_llm(OPENAI_API_KEY, temperature, r_llm, user_llm_path)    
    #if ( pinecone_index_name or chroma_collection_name ) and embeddings and llm:
        session_name = pinecone_index_name + chroma_collection_name + hist_fn
        if usage != 'Chat':
            if r_ingest.lower() == 'yes':
                files = st.file_uploader(
                    'Upload Files', accept_multiple_files=True)
                if files:
                    save_file(files)
                    all_texts, n_texts = load_files()
                    if usage == 'RAG':
                        docsearch = ingest(all_texts, use_pinecone, embeddings, pinecone_index_name,
                                    chroma_collection_name, persist_directory)
                    docsearch_ready = True
            else:
                st.write(
                    'No data is to be ingested. Make sure the Pinecone index or Chroma collection name you provided contains data.')
                docsearch, n_texts = setup_docsearch(use_pinecone, pinecone_index_name,
                                                    embeddings, chroma_collection_name, persist_directory)
                docsearch_ready = True
        else:
            docsearch_ready = True
    if docsearch_ready:
        prompt = setup_prompt(r_llm, usage)      
        #if usage == 'RAG' or usage == 'Chat':
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        if usage == 'RAG':
            # number of sources (split-documents when ingesting files); default is 4
            k = min([20, n_texts])
            retriever = setup_retriever(docsearch, k)
            CRqa = ConversationalRetrievalChain.from_llm(
                    llm, 
                    chain_type="stuff",
                    retriever=retriever, 
                    memory=memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={'prompt': prompt},
                    )
        elif usage == 'Chat':   
            CRqa = LLMChain(
                    llm=llm, 
                    prompt=prompt,                        
                    )
        elif usage == 'Task':                
            CRqa = load_qa_chain(
                    llm=llm, 
                    chain_type="stuff",
                    prompt=prompt
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
            if usage == 'RAG':
                reply, source = get_response(query, chat_history, CRqa)
            elif usage == 'Chat':
                reply = CRqa({"question": query, "chat_history": chat_history, "return_only_outputs": True})
                reply = reply['text']
            elif usage == 'Task':
                reply = []
                for a_text in all_texts:
                    output_text = CRqa.run(input_documents=[a_text], question=query )                    
                    reply.append ( output_text )
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

        if usage == 'RAG' and reply and source:
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
        st.text_area('Chat records in ascending order:', value=all_chat_history_str, height=250, label_visibility='collapsed')      
if __name__ == '__main__':
    main(pinecone_index_name, chroma_collection_name, persist_directory,
         docsearch_ready, directory_name)
