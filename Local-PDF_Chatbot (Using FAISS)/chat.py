import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from model_map import load_llm

DB_faiss_path = "vector/db_faiss"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})

db = FAISS.load_local(DB_faiss_path, embeddings)

# Define the Streamlit app
st.set_page_config(page_title="Multi-Doc Chatbot", page_icon=":robot_face:")

# Streamlit App Title
st.title("Document Chat-bot :robot_face::books:")


def initailize():    
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.subheader('Model and parameter Selection')
    # Select a Quantised model
    model = st.selectbox(
        "Select a Quantised model",
        ("Select any one from the llm's","Vicuna-13b 5-bit","mistral-7b-instruct 4-bit","mistral-7b-instruct 5-bit","llama-2-7b 4-bit","llama-2-7b-chat 5-bit")
    )
    # # Adjust temperature and max_length based on user input
    temperature = st.slider('Temperature', min_value=0.01, max_value=5.0,step=0.01)
    max_length = st.slider('Max Length', min_value=10, max_value=512,step=8)

    initailize()

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def generate_response(prompt):

    qa_chain = RetrievalQA.from_chain_type(
                                        # llm=load_llm(model,temperature,max_length),
                                        llm = load_llm(model),
                                        chain_type='stuff',
                                        retriever=db.as_retriever(search_kwargs={'k': 2}),
                                        return_source_documents=True)

    # Process the user's query
    llm_response = qa_chain(prompt)

    # Extract the response
    response = llm_response['result']
    return response

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)    

if prompt := st.chat_input("Enter your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating Response.."):
            response = generate_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


