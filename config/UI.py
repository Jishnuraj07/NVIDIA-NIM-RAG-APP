import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain_nvidia_ai_endpoints import NVIDIARerank
from nemoguardrails import LLMRails, RailsConfig
import asyncio

import nest_asyncio

nest_asyncio.apply()


# Initialize the reranker (NVIDIA reranker)
compressor = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3", base_url="http://localhost:11737/v1")

# Define the Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=docsearch.as_retriever()  # Ensure 'docsearch' is defined earlier in your code
)

# Initialize NeMo Guardrails
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Function to rerank query results and generate a response for the /complete endpoint
def rerank_and_generate_response(user_query: str):
    # Perform reranking using NVIDIA reranker
    documents = compression_retriever.get_relevant_documents(user_query)
    context_str = "\n".join([doc.page_content for doc in documents[:3]])  # Get top 3 documents

    # Generate a prompt for the assistant using the reranked documents
    prompt = f"""
    You are an assistant that answers queries using CUDA-related knowledge.
    Based on the following relevant documents, answer the user's query.
    
    Relevant Documents (Top 3): 
    {context_str[:1200]}  # Ensuring the context doesn't exceed 1200 tokens

    User Query: {user_query}
    
    Answer (Max 750 tokens):
    """

    # Generate response using NeMo Guardrails
    response = rails.generate(messages=[
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": prompt}
    ])

    # Return the assistant's response and the context used
    return {
        "answer": response['content'][:750],  # Ensuring answer doesn't exceed 750 tokens
        "contexts": context_str[:1200]  # Ensuring contexts don't exceed 1200 tokens
    }

# Function to reset the session for the /reset endpoint by reinitializing the LLMRails object
def reset_conversation():
    global rails  # Use global to reset the rails object
    rails = LLMRails(config)  # Reinitialize the LLMRails to simulate session reset
    return {"status": "Conversation reset successfully."}

# Streamlit UI
st.title("CUDA Query Assistant")

# User input for query
user_query = st.text_input("Enter your CUDA-related query:")

# Submit button to query the assistant
if st.button("Submit Query"):
    if user_query:
        result = rerank_and_generate_response(user_query)
        st.subheader("Answer:")
        st.write(result["answer"])
        
        st.subheader("Contexts:")
        st.write(result["contexts"])
    else:
        st.warning("Please enter a query.")

# Button to reset the conversation
if st.button("Reset Conversation"):
    reset_status = reset_conversation()
    st.success(reset_status["status"])

