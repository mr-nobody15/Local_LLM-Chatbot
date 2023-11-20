from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st
import textwrap
import os

DB_faiss_path = "vector/db_faiss"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})

db = FAISS.load_local(DB_faiss_path, embeddings)

# Define a dictionary mapping user-friendly names to model file names
MODEL_MAPPING = {
    "llama-2-7b 4-bit":"llama-2-7b.ggmlv3.q4_K_M.bin",
    "llama-2-7b-chat 5-bit" : "llama-2-7b-chat.ggmlv3.q5_K_M.bin",
    "Vicuna-13b 5-bit": "ggml-vic13b-q5_1.bin",
    "mistral-7b-instruct 4-bit" : "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "mistral-7b-instruct 5-bit" : "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
    # Add more mappings as needed
}

# Loading the model
def load_llm(model_name,temp = 0.1,max_tokens = 512):
    if model_name in MODEL_MAPPING:
        model_file = MODEL_MAPPING[model_name]
        model_path = f"./Models/{model_file}"
        # model_path = f"D:/Internship/Chatbot_using-Llama-2_4bit/models/{model_file}"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model '{model_name}' not found in the 'models' folder.")
        
        llm = CTransformers(
            model=model_path,
            model_type="llama",
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=0.9
        )
        
        return llm
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")

def get_initial_message():
    messages=[
            {"role": "system", "content": "You are a helpful Automated Guide. Who answers brief questions from the user."},
            {"role": "user", "content": "I want to learn AI"},
            {"role": "assistant", "content": "That's a very good choice, What do you want to learn?"}
        ]
    return messages

def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages

def get_chatgpt_response(messages, model, query=None):  
    
    # Add 'query' as a parameter with a default value of None
    # Implement the code to interact with the GPT-based model here
    # Return the generated response

    qa_chain = RetrievalQA.from_chain_type(llm=load_llm(model),
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True)
    
    if query:
        messages.append({"role": "user", "content": query})

        query_text = query

        # Call Langchain with the query_text
        llm_response = qa_chain([{"content": query_text}])

        # Extract the response
        response = llm_response['result']
    
    # # Combine the messages into a single string
    # message_text = "\n".join([message["content"] for message in messages])

    # # Call Langchain with the combined message_text
    # llm_response = qa_chain([{"content": message_text}])

    # # Extract the response
    # response = llm_response['result']

        # Define the maximum number of tokens to display
        max_tokens_to_display = 512  # Adjust this value as needed

        # Check if the response exceeds the maximum token limit
        if len(response.split()) > max_tokens_to_display:
            # Split the response into tokens
            tokens = response.split()
            
            # Truncate the tokens to the maximum limit
            truncated_tokens = tokens[:max_tokens_to_display]
            
            # Join the truncated tokens back into a string
            truncated_response = ' '.join(truncated_tokens)
            
            # Wrap and display the truncated response
            wrapped_response = textwrap.fill(truncated_response, width=110)
            st.markdown("## Response:")
            st.write(wrapped_response)
        else:
            # The response is within the token limit, so display it as is
            wrapped_response = textwrap.fill(response, width=110)
            st.markdown("## Response:")
            st.write(wrapped_response)
