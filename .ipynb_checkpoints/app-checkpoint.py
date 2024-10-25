import sys
import streamlit as st
import requests
from flask import Flask, request, jsonify
import threading
import itertools
import logging
from nemoguardrails import LLMRails, RailsConfig
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_nvidia_ai_endpoints import NVIDIARerank, NVIDIAEmbeddings, ChatNVIDIA
import re


# Configure logging to capture detailed information for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure that st.set_page_config() is the first Streamlit command
st.set_page_config(page_title="🧠 Edge AI: CUDA Knowledge Assistant", layout="wide")

# Add custom module path if needed (adjust the path as per your environment)
sys.path.insert(0, '/mnt/lustre/hackathons/hack_teams/hack_team_16/workspace/Jishnu')

app = Flask(__name__)
memory = 0  # Global memory variable for tracking state


# Initialize NeMo Guardrails with your configuration
config_path = "./config"  # Adjust the path to your Guardrails configuration
try:
    config = RailsConfig.from_path(config_path)
    rails = LLMRails(config)
    logging.info("NeMo Guardrails initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize NeMo Guardrails: {e}")
    logging.error(f"Failed to initialize NeMo Guardrails: {e}")
    st.stop()

# Initialize the NVIDIA-based LLM
try:
    llm = ChatNVIDIA(
        base_url="http://localhost:8000/v1",  # Adjust the base URL as per your setup
        model="meta/llama-3.1-8b-instruct",   # Replace with your desired model
        temperature=0,
        max_tokens=1000
    )
    logging.info("NVIDIA-based LLM initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize NVIDIA-based LLM: {e}")
    logging.error(f"Failed to initialize NVIDIA-based LLM: {e}")
    st.stop()

# Initialize the embeddings model
try:
    embeddings_model = NVIDIAEmbeddings(
        base_url="http://localhost:11022/v1",  # Adjust the base URL as per your setup
        model='nvidia/nv-embedqa-e5-v5'        # Replace with your desired embeddings model
    )
    logging.info("NVIDIA Embeddings model initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize NVIDIA Embeddings model: {e}")
    logging.error(f"Failed to initialize NVIDIA Embeddings model: {e}")
    st.stop()

# Load FAISS vector store
embedding_path = "./embed/"  # Adjust the path to your FAISS embeddings
try:
    docsearch = FAISS.load_local(
        folder_path=embedding_path,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True  # Ensure you trust the source
    )
    logging.info("FAISS vector store loaded successfully.")
except Exception as e:
    st.error(f"Failed to load FAISS vector store: {e}")
    logging.error(f"Failed to load FAISS vector store: {e}")
    st.stop()

# Initialize the NVIDIA reranker
try:
    compressor = NVIDIARerank(
        model="nvidia/nv-rerankqa-mistral-4b-v3",  # Replace with your desired reranker model
        base_url="http://localhost:11737/v1"      # Adjust the base URL as per your setup
    )
    logging.info("NVIDIA Reranker initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize NVIDIA Reranker: {e}")
    logging.error(f"Failed to initialize NVIDIA Reranker: {e}")
    st.stop()

# Define the Contextual Compression Retriever
try:
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=docsearch.as_retriever()
    )
    logging.info("Contextual Compression Retriever initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize Contextual Compression Retriever: {e}")
    logging.error(f"Failed to initialize Contextual Compression Retriever: {e}")
    st.stop()

# Define supported languages (all in lowercase for consistency)
SUPPORTED_LANGUAGES = {"french", "spanish", "german", "italian", "japanese", "portuguese", "chinese", "korean"}


@app.route('/complete', methods=['POST'])
def complete_endpoint():
    global memory
    data = request.get_json()
    question = data.get('question')
    
    if question:
        # Split input string into words
        words = question.split()
        # Sort words by length and join them back together
        answer = ' '.join(sorted(words, key=len))
        # Generate all permutations of the question string words
        contexts = [' '.join(p) for p in itertools.permutations(words)]
        memory = 1  # Update memory flag
        return jsonify({'answer': answer, 'contexts': contexts})
    
    return jsonify({'error': 'Invalid input'}), 400

@app.route('/reset', methods=['POST'])
def reset_endpoint():
    global memory
    data = request.get_json()
    flag = data.get('reset')
    
    if flag == 1:
        memory = 0
        return jsonify({'reset_response': 'success'})
    else:
        return jsonify({'reset_response': 'failure'})
    
    return jsonify({'error': 'Invalid input'}), 400

def run_flask_app():
    """Run the Flask app on a separate thread"""
    app.run(port=5001, debug=False, use_reloader=False, host='0.0.0.0')

# Start the Flask app in a separate thread
threading.Thread(target=run_flask_app, daemon=True).start()

# NVIDIA-based LLM initialization (use Flask `/complete` endpoint)
def get_answer_from_flask(question):
    try:
        response = requests.post('http://localhost:5001/complete', json={'question': question})
        if response.status_code == 200:
            return response.json()
        else:
            return {'answer': 'Error occurred', 'contexts': []}
    except Exception as e:
        return {'answer': f'Error: {str(e)}', 'contexts': []}

def reset_memory_in_flask():
    try:
        response = requests.post('http://localhost:5001/reset', json={'reset': 1})
        if response.status_code == 200:
            return response.json()
        else:
            return {'reset_response': 'failure'}
    except Exception as e:
        return {'reset_response': f'Error: {str(e)}'}

# Initialize NeMo Guardrails with your configuration
config_path = "./config"  # Adjust the path to your Guardrails configuration
try:
    config = RailsConfig.from_path(config_path)
    rails = LLMRails(config)
    logging.info("NeMo Guardrails initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize NeMo Guardrails: {e}")
    logging.error(f"Failed to initialize NeMo Guardrails: {e}")
    st.stop()

# NVIDIA-based LLM initialization (use Flask `/complete` endpoint)
def get_answer_from_flask(question):
    try:
        response = requests.post('http://localhost:5001/complete', json={'question': question})
        if response.status_code == 200:
            return response.json()
        else:
            return {'answer': 'Error occurred', 'contexts': []}
    except Exception as e:
        return {'answer': f'Error: {str(e)}', 'contexts': []}

def reset_memory_in_flask():
    try:
        response = requests.post('http://localhost:5001/reset', json={'reset': 1})
        if response.status_code == 200:
            return response.json()
        else:
            return {'reset_response': 'failure'}
    except Exception as e:
        return {'reset_response': f'Error: {str(e)}'}


def detect_user_intent(user_query: str) -> str:
    """
    Detects the user's intent based on their query using NeMo Guardrails.
    Returns one of the predefined intents.
    """
    detect_intent_prompt = f"""
Your task is to classify the user's query into one of the following specific intents. 
You MUST return ONLY the intent name from this list, enclosed in double quotes:

**Possible Intents:**
- "code_generation"
- "code_debugging"
- "cuda_explanation"
- "general_question"

**Classification Criteria:**
- If the query mentions phrases like "write code", "generate code", "provide code", or "create a program", classify it as "code_generation".
- If the query asks for help fixing or debugging code, classify it as "code_debugging".
- If the query asks to explain a concept, classify it as "cuda_explanation".
- If the query is high-level or doesn't fit the above categories, classify it as "general_question".

**Examples:**
- "Write a CUDA program to multiply matrices." → "code_generation"
- "Explain what CUDA unified memory is." → "cuda_explanation"
- "Why is my CUDA kernel crashing?" → "code_debugging"
- "What is CUDA?" → "general_question"

**Important:** 
- Return ONLY the name of the detected intent enclosed in double quotes.
- DO NOT generate a full response or explanation.
- If the query does not fit any of the above intents, respond with "not_cuda_related".

User Query: {user_query}
"""

    try:
        messages = [
            {"role": "system", "content": "You are an assistant that classifies queries into specific intents."},
            {"role": "user", "content": detect_intent_prompt}
        ]

        response = rails.generate(messages=messages)
        detected_intent = response['content'].strip().lower().replace('"', '')  # Remove quotes and lowercase

        valid_intents = {'code_generation', 'code_debugging', 'cuda_explanation', 'general_question'}

        if detected_intent not in valid_intents:
            st.warning(f"Detected intent is invalid ('{detected_intent}'). Falling back to 'general_question'")
            detected_intent = 'general_question'

        logging.info(f"Detected intent: {detected_intent} for query: {user_query}")
        return detected_intent

    except Exception as e:
        st.error(f"Error detecting intent for query '{user_query}': {e}")
        logging.error(f"Error detecting intent for query '{user_query}': {e}")
        return 'general_question'

def handle_general_question(user_query: str, detected_intent: str) -> dict:
    """
    Handles general CUDA-related questions and translation requests.
    Returns a dictionary with the assistant's answer and relevant contexts.
    """
    # Define intent-specific messages if needed
    intent_messages = {
        'code_generation': "Please generate CUDA code for the following query.",
        'code_debugging': "Please help debug the following CUDA code issue.",
        'cuda_explanation': "Please explain the following CUDA concept.",
        'general_question': "Please answer the following general CUDA-related question."
    }

    assistant_message = intent_messages.get(detected_intent, "Please assist with the following query.")
    logging.info(f"Assistant message based on intent '{detected_intent}': {assistant_message}")

    # Retrieve and rerank documents
    try:
        documents = compression_retriever.invoke(user_query)
        logging.info(f"Retrieved {len(documents)} documents for query: {user_query}")
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        logging.error(f"Error retrieving documents: {e}")
        return {
            "answer": "Sorry, I couldn't retrieve relevant documents at this time.",
            "contexts": []
        }

    if not documents:
        logging.info("No relevant documents found.")
        return {
            "answer": "Sorry, I couldn't find any relevant documents.",
            "contexts": []
        }

    # Limit to top 3 documents
    top_documents = documents[:3]
    logging.info(f"Using top {len(top_documents)} documents for response generation.")

    # Extract page content from documents
    contexts = [doc.page_content for doc in top_documents]

    # Truncate contexts to ensure total tokens do not exceed 1200
    total_tokens = 0
    max_tokens = 1200
    truncated_contexts = []
    for context in contexts:
        context_tokens = len(context.split())
        if total_tokens + context_tokens <= max_tokens:
            truncated_contexts.append(context)
            total_tokens += context_tokens
        else:
            # Truncate the context to fit into the remaining tokens
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 0:
                truncated_context = ' '.join(context.split()[:remaining_tokens])
                truncated_contexts.append(truncated_context)
                logging.info("Truncated context to fit token limit.")
            break  # No more contexts can be added

    logging.info(f"Total tokens in context: {total_tokens}")

    # Prepare the user message content
    user_message_content = f"""You are an assistant that answers queries using CUDA-related knowledge.
Based on the following relevant documents, answer the user's query.

User's Query:
{user_query}

Relevant Documents (Top 3):
""" + "\n---\n".join(truncated_contexts) + """
Please keep in mind the following restrictions:
● answer: This is the RAG pipeline’s main response to the query. The response shouldn’t have more than 750 tokens.
● contexts: The top documents that were retrieved for the query and utilized for generating the response. The contexts shall not exceed more than 1200 tokens. It can be a list of a maximum of three documents.
"""

    logging.info(f"User message content prepared for intent '{detected_intent}'.")

    # Incorporate conversation history into messages to maintain context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in st.session_state.conversation[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})

    messages.append({"role": "user", "content": user_message_content})
    messages.append({"role": "assistant", "content": assistant_message})

    logging.info("Conversation history added to messages.")

    # Detect if the user query is a translation request
    translation_keywords = ['translate', 'translation']
    is_translation = any(keyword in user_query.lower() for keyword in translation_keywords)

    if is_translation:
        # Extract target language from the query using regex
        match = re.search(r'\b(to|into)\s+(\w+)', user_query.lower())
        if match:
            target_language = match.group(2).capitalize()
            if target_language.lower() not in SUPPORTED_LANGUAGES:
                answer = f"I'm sorry, I don't support translating into {target_language}. Supported languages are: {', '.join([lang.capitalize() for lang in SUPPORTED_LANGUAGES])}."
                logging.info(f"Unsupported target language: {target_language}")
            else:
                # Prepare the translation prompt
                translation_prompt = f"Translate the following CUDA-related content into {target_language}:\n\n{user_query}"
                # Incorporate conversation history into messages to maintain context
                translation_messages = [{"role": "system", "content": "You are a helpful assistant."}]
                for exchange in st.session_state.conversation[-10:]:
                    translation_messages.append({"role": "user", "content": exchange["user"]})
                    translation_messages.append({"role": "assistant", "content": exchange["assistant"]})
                translation_messages.append({"role": "user", "content": translation_prompt})
                translation_messages.append({"role": "assistant", "content": ""})  # The assistant will fill this

                try:
                    # Generate translation response using NeMo Guardrails
                    translation_response = rails.generate(messages=translation_messages)
                    translated_text = translation_response.get('content', '').strip()
                    answer = translated_text
                    logging.info(f"Translated answer into {target_language}: {translated_text[:100]}...")
                except Exception as e:
                    st.error(f"Error generating translation: {e}")
                    logging.error(f"Error generating translation: {e}")
                    answer = "Sorry, I couldn't generate a translation at this time."
        else:
            answer = "I'm sorry, I couldn't detect the target language for translation. Please specify the language you'd like the content translated into."
            logging.info("Target language for translation not detected.")
    else:
        # Proceed with generating general answer
        try:
            # Generate response using NeMo Guardrails
            response = rails.generate(messages=messages)
            answer = response.get('content', '').strip()
            logging.info(f"Generated answer: {answer[:100]}...")  # Log first 100 characters
        except Exception as e:
            st.error(f"Error generating response: {e}")
            logging.error(f"Error generating response: {e}")
            answer = "Sorry, I couldn't generate a response at this time."

    # Ensure the answer does not exceed 750 tokens
    answer_tokens = len(answer.split())
    if answer_tokens > 750:
        answer = ' '.join(answer.split()[:750])
        st.warning("Answer truncated to 750 tokens.")
        logging.warning("Answer truncated to 750 tokens.")

    # Return the assistant's response along with contexts
    return {
        "answer": answer,
        "contexts": truncated_contexts
    }

def handle_cuda_explanation(user_query: str) -> dict:
    """
    Handles CUDA explanation intents.
    Returns a dictionary with the assistant's explanation and relevant contexts.
    """
    # Retrieve and rerank documents
    try:
        documents = compression_retriever.invoke(user_query)
        logging.info(f"Retrieved {len(documents)} documents for query: {user_query}")
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        logging.error(f"Error retrieving documents: {e}")
        return {
            "answer": "Sorry, I couldn't retrieve relevant documents at this time.",
            "contexts": []
        }

    if not documents:
        logging.info("No relevant documents found.")
        return {
            "answer": "Sorry, I couldn't find any relevant documents.",
            "contexts": []
        }

    # Limit to top 3 documents
    top_documents = documents[:3]
    logging.info(f"Using top {len(top_documents)} documents for explanation generation.")

    # Extract page content from documents
    contexts = [doc.page_content for doc in top_documents]

    # Truncate contexts to ensure total tokens do not exceed 1200
    total_tokens = 0
    max_tokens = 1200
    truncated_contexts = []
    for context in contexts:
        context_tokens = len(context.split())
        if total_tokens + context_tokens <= max_tokens:
            truncated_contexts.append(context)
            total_tokens += context_tokens
        else:
            # Truncate the context to fit into the remaining tokens
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 0:
                truncated_context = ' '.join(context.split()[:remaining_tokens])
                truncated_contexts.append(truncated_context)
                logging.info("Truncated context to fit token limit.")
            break  # No more contexts can be added

    logging.info(f"Total tokens in context: {total_tokens}")

    # Prepare the explanation prompt with the provided instructions
    user_message_content = f"""You are an assistant that explains CUDA concepts clearly and concisely.
Ensure that the explanation is accurate, contextually appropriate, and derived from reliable sources.

**Instructions:**
- Provide a clear and detailed explanation of the CUDA concept mentioned.
- Use examples if necessary to enhance understanding.
- Avoid excessive technical jargon to ensure clarity.

User Query: {user_query}
Explanation (Max 300 tokens):
"""

    logging.info(f"User message content prepared for CUDA explanation.")

    # Incorporate conversation history into messages to maintain context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in st.session_state.conversation[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})

    messages.append({"role": "user", "content": user_message_content})
    messages.append({"role": "assistant", "content": "Please provide the detailed explanation."})

    try:
        # Generate explanation response using NeMo Guardrails
        response = rails.generate(messages=messages)
        explanation = response.get('content', '').strip()
        logging.info(f"Generated explanation: {explanation[:100]}...")  # Log first 100 characters

        # Ensure the explanation does not exceed 300 tokens
        explanation_tokens = len(explanation.split())
        if explanation_tokens > 300:
            explanation = ' '.join(explanation.split()[:300])
            st.warning("Explanation truncated to 300 tokens.")
            logging.warning("Explanation truncated to 300 tokens.")

        # Return the assistant's explanation response along with contexts
        return {
            "answer": explanation,
            "contexts": truncated_contexts
        }
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        logging.error(f"Error generating explanation: {e}")
        return {
            "answer": "Sorry, I couldn't generate an explanation at this time.",
            "contexts": truncated_contexts
        }

def generate_cuda_code(user_query: str) -> dict:
    """
    Handles CUDA code generation intents.
    Returns a dictionary with the assistant's code and relevant contexts.
    """
    # Retrieve and rerank documents
    try:
        documents = compression_retriever.invoke(user_query)
        logging.info(f"Retrieved {len(documents)} documents for query: {user_query}")
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        logging.error(f"Error retrieving documents: {e}")
        return {
            "answer": "Sorry, I couldn't retrieve relevant documents at this time.",
            "contexts": []
        }

    if not documents:
        logging.info("No relevant documents found.")
        return {
            "answer": "Sorry, I couldn't find any relevant documents.",
            "contexts": []
        }

    # Limit to top 3 documents
    top_documents = documents[:3]
    logging.info(f"Using top {len(top_documents)} documents for code generation.")

    # Extract page content from documents
    contexts = [doc.page_content for doc in top_documents]

    # Truncate contexts to ensure total tokens do not exceed 1200
    total_tokens = 0
    max_tokens = 1200
    truncated_contexts = []
    for context in contexts:
        context_tokens = len(context.split())
        if total_tokens + context_tokens <= max_tokens:
            truncated_contexts.append(context)
            total_tokens += context_tokens
        else:
            # Truncate the context to fit into the remaining tokens
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 0:
                truncated_context = ' '.join(context.split()[:remaining_tokens])
                truncated_contexts.append(truncated_context)
                logging.info("Truncated context to fit token limit.")
            break  # No more contexts can be added

    logging.info(f"Total tokens in context: {total_tokens}")

    # Prepare the code generation prompt with the provided instructions
    user_message_content = f"""You are an assistant that generates CUDA code based on user queries.
Ensure that the code follows best practices and performance optimization techniques as outlined in CUDA documentation.

**Instructions:**
- Generate efficient CUDA code for the specified task.
- Follow CUDA performance optimization guidelines, including:
  - Memory coalescing
  - Avoiding warp divergence
  - Minimizing shared memory usage
- If specific requirements (e.g., memory management, kernel behavior) are provided, address them explicitly in the code.
- If unsure, return the best possible code and mention areas that need clarification.
- Utilize conversation history to resolve references and maintain context.

User Query: {user_query}
Generated CUDA Code (Max 500 tokens):
"""

    logging.info(f"User message content prepared for CUDA code generation.")

    # Incorporate conversation history into messages to maintain context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in st.session_state.conversation[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})

    messages.append({"role": "user", "content": user_message_content})
    messages.append({"role": "assistant", "content": "Please provide the CUDA code for matrix multiplication."})

    try:
        # Generate CUDA code response using NeMo Guardrails
        response = rails.generate(messages=messages)
        code = response.get('content', '').strip()
        logging.info(f"Generated CUDA code: {code[:100]}...")  # Log first 100 characters

        # Ensure the code does not exceed 500 tokens
        code_tokens = len(code.split())
        if code_tokens > 500:
            code = ' '.join(code.split()[:500])
            st.warning("CUDA code truncated to 500 tokens.")
            logging.warning("CUDA code truncated to 500 tokens.")

        # Return the assistant's code response along with contexts
        return {
            "answer": code,
            "contexts": truncated_contexts
        }
    except Exception as e:
        st.error(f"Error generating CUDA code: {e}")
        logging.error(f"Error generating CUDA code: {e}")
        return {
            "answer": "Sorry, I couldn't generate CUDA code at this time.",
            "contexts": truncated_contexts
        }

def handle_code_debugging(user_query: str) -> dict:
    """
    Handles CUDA code debugging intents.
    Returns a dictionary with the assistant's debugging steps and relevant contexts.
    """
    # Retrieve and rerank documents
    try:
        documents = compression_retriever.invoke(user_query)
        logging.info(f"Retrieved {len(documents)} documents for query: {user_query}")
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        logging.error(f"Error retrieving documents: {e}")
        return {
            "answer": "Sorry, I couldn't retrieve relevant documents at this time.",
            "contexts": []
        }

    if not documents:
        logging.info("No relevant documents found.")
        return {
            "answer": "Sorry, I couldn't find any relevant documents.",
            "contexts": []
        }

    # Limit to top 3 documents
    top_documents = documents[:3]
    logging.info(f"Using top {len(top_documents)} documents for debugging assistance.")

    # Extract page content from documents
    contexts = [doc.page_content for doc in top_documents]

    # Truncate contexts to ensure total tokens do not exceed 1200
    total_tokens = 0
    max_tokens = 1200
    truncated_contexts = []
    for context in contexts:
        context_tokens = len(context.split())
        if total_tokens + context_tokens <= max_tokens:
            truncated_contexts.append(context)
            total_tokens += context_tokens
        else:
            # Truncate the context to fit into the remaining tokens
            remaining_tokens = max_tokens - total_tokens
            if remaining_tokens > 0:
                truncated_context = ' '.join(context.split()[:remaining_tokens])
                truncated_contexts.append(truncated_context)
                logging.info("Truncated context to fit token limit.")
            break  # No more contexts can be added

    logging.info(f"Total tokens in context: {total_tokens}")

    # Prepare the debugging prompt with the provided instructions
    user_message_content = f"""You are an assistant that assists with CUDA code debugging.
Provide debugging steps based on CUDA’s debugging best practices, including handling common issues like synchronization, memory access, and kernel crashes.

**Instructions:**
- Use tools such as printf debugging, CUDA-GDB, and Nsight Debugger.
- Recommend solutions for issues like:
  - Race conditions
  - Memory leaks
  - Warp divergence
  - Optimization bottlenecks
- Provide a step-by-step debugging process:
  1. Analyze kernel launch configuration
  2. Check memory allocations (host/device)
  3. Verify synchronization points
  4. Address shared memory access patterns

**Fallback:**
- If debugging is unclear, guide the user to provide more detailed information about the code or error logs.
- Utilize conversation history to resolve references and maintain context.

User Query: {user_query}
Debugging Steps and Solution (Max 400 tokens):
"""

    logging.info(f"User message content prepared for CUDA code debugging.")

    # Incorporate conversation history into messages to maintain context
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in st.session_state.conversation[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange["user"]})
        messages.append({"role": "assistant", "content": exchange["assistant"]})

    messages.append({"role": "user", "content": user_message_content})
    messages.append({"role": "assistant", "content": "Please provide the details of the CUDA kernel crash."})

    try:
        # Generate debugging response using NeMo Guardrails
        response = rails.generate(messages=messages)
        debugging_steps = response.get('content', '').strip()
        logging.info(f"Generated debugging steps: {debugging_steps[:100]}...")  # Log first 100 characters

        # Ensure the debugging steps do not exceed 400 tokens
        debugging_tokens = len(debugging_steps.split())
        if debugging_tokens > 400:
            debugging_steps = ' '.join(debugging_steps.split()[:400])
            st.warning("Debugging steps truncated to 400 tokens.")
            logging.warning("Debugging steps truncated to 400 tokens.")

        # Return the assistant's debugging response along with contexts
        return {
            "answer": debugging_steps,
            "contexts": truncated_contexts
        }
    except Exception as e:
        st.error(f"Error generating debugging steps: {e}")
        logging.error(f"Error generating debugging steps: {e}")
        return {
            "answer": "Sorry, I couldn't generate debugging steps at this time.",
            "contexts": truncated_contexts
        }

def main():
    """
    The main function that runs the Streamlit application.
    Handles user interactions, detects intents, and displays responses.
    """
    st.title("🧠 **Edge AI: CUDA Knowledge Assistant**")

    # Initialize session state variables if they don't exist
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    # Form for user query input
    with st.form(key='query_form'):
        user_query = st.text_input("Enter your CUDA-related question:", "")
        submit_button = st.form_submit_button(label='Get Answer')

    if submit_button and user_query.strip() != "":
        # Store the current query in session state
        st.session_state.current_query = user_query

        # Fetch the answer and contexts from the Flask `/complete` endpoint
        with st.spinner("Processing..."):
            result = get_answer_from_flask(user_query)
        
        # Display the answer
        st.subheader("💡 **Answer:**")
        st.write(result['answer'])

        # Display the contexts
        if result['contexts']:
            st.subheader("📄 **Generated Contexts:**")
            for idx, context in enumerate(result['contexts']):
                with st.expander(f"📑 Context {idx + 1}"):
                    st.write(context)

        # Append to conversation history
        st.session_state.conversation.append({
            "user": user_query,
            "assistant": result['answer']
        })

    # Reset button for memory in Flask app
    if st.button("🔄 Reset Memory"):
        reset_result = reset_memory_in_flask()
        st.success(reset_result.get('reset_response', 'Error resetting memory'))

    # Display conversation history in the sidebar
    if st.session_state.conversation:
        st.sidebar.subheader("🗨️ **Conversation History**")
        for exchange in reversed(st.session_state.conversation[-5:]):  # Show last 5 exchanges
            st.sidebar.markdown(f"**User:** {exchange['user']}")
            st.sidebar.markdown(f"**Assistant:** {exchange['assistant']}")
            st.sidebar.markdown("---")

    # Reset conversation button
    if st.button("🔄 Reset Conversation"):
        st.session_state.conversation = []
        st.session_state.current_query = ""
        st.success("✅ Conversation reset successfully.")

if __name__ == "__main__":
    main()