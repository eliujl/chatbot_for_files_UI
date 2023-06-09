
This is a chatbot that uses Langchain's Conversational Retrieval Chain to generate responses to user input. The chatbot can ingest files and use Pinecone (Pinecone API key required) or Chroma vector stores (no API key required) to retrieve relevant documents for generating responses. OpenAI's API key is also required. The UI is based on Streamlit.

## Fun fact
This README file is generated by this app after ingesting this python file. See the screenshot below.

## Installation

To install the required packages, run:

```
pip install -r requirements.txt
```

## Usage

To run the chatbot, run:

```
streamlit run chatbot_for_files_st.py
```

The chatbot will prompt the user for inputs and generate a response based on user's question and the chat history.

## Ingesting Files

To ingest files, select "Yes" when prompted and upload the files. The chatbot will split the files into smaller documents and ingest them into the vector store.

## Using Pinecone

To use Pinecone, select "Yes" when prompted and enter the name of the Pinecone index. Make sure to set the `PINECONE_API_KEY` and `PINECONE_API_ENV` environment variables.

## Using Chroma

To use Chroma, enter the name of the Chroma collection when prompted. The chatbot will create a Chroma vector store in the `persist_directory` specified in the code.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Screenshot
![chatbot](https://github.com/eliujl/chatbot_for_files_UI/assets/8711788/d8b6dcdb-0777-4d73-abea-ef651b41f0df)


