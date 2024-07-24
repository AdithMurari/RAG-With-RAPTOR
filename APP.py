from langchain.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import pickle
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from Qexp import expand_query

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
)

# Connect to Milvus
connections.connect(host='127.0.0.1', port='19530')

# List all collections
from pymilvus import utility
collections = utility.list_collections()
print(collections)

# Get an existing collection
collection = Collection("StockMarket")

# Load the collection into memory
collection.load(replica_number=1)

dense_field_name = "dense_vector"
sparse_field_name = "sparse_vector"
text_field_name = "text"

dense_embedding_func=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",      
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)
with open('sparse_embedding_func.pkl', 'rb') as f:
    sparse_embedding_func = pickle.load(f)

sparse_search_params = {"metric_type": "IP"}
dense_search_params = {"metric_type": "IP", "params": {}}

retriever = MilvusCollectionHybridSearchRetriever(
    collection=collection,
    rerank=WeightedRanker(0.5, 0.5),
    anns_fields=[dense_field_name, sparse_field_name],  # Field names as strings
    field_embeddings=[dense_embedding_func, sparse_embedding_func],  # Embedding functions
    field_search_params=[dense_search_params, sparse_search_params],  # Search parameters
    top_k=15,
    text_field=text_field_name,  # Text field name as a string
)

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("Stock Market Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]


def get_response(user_query,chat_history):

    PROMPT_TEMPLATE = """
    Human: You are an AI assistant, and provides answers to questions by using the context provided.
    Use the following pieces of information to provide a brief answer to the question enclosed in <question> tags.

    <chat_history>
    {chat_history}
    <chat_history>

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>y

    Assistant:"""

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["chat_history","context", "question"])

    llm = ChatOpenAI()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = prompt| llm
    
    return rag_chain.invoke({
        "chat_history": chat_history,
        "context": retriever | format_docs,
        "question": user_query})



for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Ask me anything related to stock market")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.markdown(response.content)

    st.session_state.chat_history.append(AIMessage(content=response.content))