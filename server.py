import sys
import logging
from flask import Flask, request, jsonify
from nemoguardrails import LLMRails, RailsConfig
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain_nvidia_ai_endpoints import NVIDIARerank, NVIDIAEmbeddings, ChatNVIDIA
import re

app = Flask(__name__)

# Configure logging to capture detailed information for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add custom module path if needed (adjust the path as per your environment)
sys.path.insert(0, '/mnt/lustre/hackathons/hack_teams/hack_team_16/workspace/Jishnu')

# Initialize NeMo Guardrails with your configuration
config_path = "./config"  # Adjust the path to your Guardrails configuration
try:
    config = RailsConfig.from_path(config_path)
    rails = LLMRails(config)
    logging.info("NeMo Guardrails initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize NeMo Guardrails: {e}")
    sys.exit(1)

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
    logging.error(f"Failed to initialize NVIDIA-based LLM: {e}")
    sys.exit(1)

# Initialize the embeddings model
try:
    embeddings_model = NVIDIAEmbeddings(
        base_url="http://localhost:11022/v1",  # Adjust the base URL as per your setup
        model='nvidia/nv-embedqa-e5-v5'        # Replace with your desired embeddings model
    )
    logging.info("NVIDIA Embeddings model initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize NVIDIA Embeddings model: {e}")
    sys.exit(1)

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
    logging.error(f"Failed to load FAISS vector store: {e}")
    sys.exit(1)

# Initialize the NVIDIA reranker
try:
    compressor = NVIDIARerank(
        model="nvidia/nv-rerankqa-mistral-4b-v3",  # Replace with your desired reranker model
        base_url="http://localhost:11737/v1"      # Adjust the base URL as per your setup
    )
    logging.info("NVIDIA Reranker initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize NVIDIA Reranker: {e}")
    sys.exit(1)

# Define the Contextual Compression Retriever
try:
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=docsearch.as_retriever()
    )
    logging.info("Contextual Compression Retriever initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Contextual Compression Retriever: {e}")
    sys.exit(1)

# Define supported languages (all in lowercase for consistency)
SUPPORTED_LANGUAGES = {"french", "spanish", "german", "italian", "japanese", "portuguese", "chinese", "korean"}

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
- "translation"
- "general_question"

**Classification Criteria:**
- If the query mentions phrases like "write code", "generate code", "provide code", or "create a program", classify it as "code_generation".
- If the query asks for help fixing or debugging code, classify it as "code_debugging".
- If the query asks to explain a concept, classify it as "cuda_explanation".
- If the query requests translation of CUDA-related content into a different language, classify it as "translation".
- If the query is high-level or doesn't fit the above categories, classify it as "general_question".

**Examples:**
- "Write a CUDA program to multiply matrices." → "code_generation"
- "Explain what CUDA unified memory is." → "cuda_explanation"
- "Why is my CUDA kernel crashing?" → "code_debugging"
- "What is CUDA?" → "general_question"
- "Translate this CUDA explanation into French." → "translation"
- "Can you translate my CUDA code comments to Spanish?" → "translation"
- "Please translate the following CUDA code into German." → "translation"
- "I need this CUDA function translated into Italian." → "translation"
- "Translate it into French." → "translation"
- "Can you help me translate my CUDA script to Japanese?" → "translation"

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

        valid_intents = {'code_generation', 'code_debugging', 'cuda_explanation', 'translation', 'general_question'}

        if detected_intent not in valid_intents:
            logging.warning(f"Detected intent is invalid ('{detected_intent}'). Falling back to 'general_question'")
            detected_intent = 'general_question'

        logging.info(f"Detected intent: {detected_intent} for query: {user_query}")
        return detected_intent

    except Exception as e:
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
        'general_question': "Please answer the following general CUDA-related question.",
        'translation': "Please translate the following CUDA-related content."
    }

    assistant_message = intent_messages.get(detected_intent, "Please assist with the following query.")
    logging.info(f"Assistant message based on intent '{detected_intent}': {assistant_message}")

    # Retrieve and rerank documents
    try:
        documents = compression_retriever.invoke(user_query)
        logging.info(f"Retrieved {len(documents)} documents for query: {user_query}")
    except Exception as e:
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
""" + "\n---\n".join(truncated_contexts)

    logging.info(f"User message content prepared for intent '{detected_intent}'.")

    # Incorporate conversation history into messages to maintain context
    # Since Flask is stateless, conversation history management is handled via client-side
    conversation_history = request.json.get('conversation', []) if request.is_json else []

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in conversation_history[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange.get("user", "")})
        messages.append({"role": "assistant", "content": exchange.get("assistant", "")})

    messages.append({"role": "user", "content": user_message_content})
    messages.append({"role": "assistant", "content": assistant_message})

    logging.info("Conversation history added to messages.")

    # Detect if the user query is a translation request
    if detected_intent == 'translation':
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
                for exchange in conversation_history[-10:]:
                    translation_messages.append({"role": "user", "content": exchange.get("user", "")})
                    translation_messages.append({"role": "assistant", "content": exchange.get("assistant", "")})
                translation_messages.append({"role": "user", "content": translation_prompt})
                translation_messages.append({"role": "assistant", "content": ""})  # The assistant will fill this

                try:
                    # Generate translation response using NeMo Guardrails
                    translation_response = rails.generate(messages=translation_messages)
                    translated_text = translation_response.get('content', '').strip()
                    answer = translated_text
                    logging.info(f"Translated answer into {target_language}: {translated_text[:100]}...")
                except Exception as e:
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
            logging.error(f"Error generating response: {e}")
            answer = "Sorry, I couldn't generate a response at this time."

    # Ensure the answer does not exceed 750 tokens
    answer_tokens = len(answer.split())
    if answer_tokens > 750:
        answer = ' '.join(answer.split()[:750])
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
    explanation_prompt = f"""You are an assistant that explains CUDA concepts clearly and concisely.
Ensure that the explanation is accurate, contextually appropriate, and derived from reliable sources.

User Query:
{user_query}

Relevant Documents (Top 3):
""" + "\n---\n".join(truncated_contexts)

    logging.info(f"User message content prepared for CUDA explanation.")

    # Incorporate conversation history into messages to maintain context
    # Since Flask is stateless, conversation history management is handled via client-side
    conversation_history = request.json.get('conversation', []) if request.is_json else []

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in conversation_history[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange.get("user", "")})
        messages.append({"role": "assistant", "content": exchange.get("assistant", "")})

    messages.append({"role": "user", "content": explanation_prompt})
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
            logging.warning("Explanation truncated to 300 tokens.")

        # Return the assistant's explanation response along with contexts
        return {
            "answer": explanation,
            "contexts": truncated_contexts
        }
    except Exception as e:
        logging.error(f"Error generating explanation: {e}")
        return {
            "answer": "Sorry, I couldn't generate an explanation at this time.",
            "contexts": []
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
    code_generation_prompt = f"""You are an assistant that generates CUDA code based on user queries.
Ensure that the code follows best practices and performance optimization techniques as outlined in CUDA documentation.

User Query:
{user_query}

Relevant Documents (Top 3):
""" + "\n---\n".join(truncated_contexts)

    logging.info(f"User message content prepared for CUDA code generation.")

    # Incorporate conversation history into messages to maintain context
    # Since Flask is stateless, conversation history management is handled via client-side
    conversation_history = request.json.get('conversation', []) if request.is_json else []

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in conversation_history[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange.get("user", "")})
        messages.append({"role": "assistant", "content": exchange.get("assistant", "")})

    messages.append({"role": "user", "content": code_generation_prompt})
    messages.append({"role": "assistant", "content": assistant_message})

    try:
        # Generate CUDA code response using NeMo Guardrails
        response = rails.generate(messages=messages)
        code = response.get('content', '').strip()
        logging.info(f"Generated CUDA code: {code[:100]}...")  # Log first 100 characters

        # Ensure the code does not exceed 500 tokens
        code_tokens = len(code.split())
        if code_tokens > 500:
            code = ' '.join(code.split()[:500])
            logging.warning("CUDA code truncated to 500 tokens.")

        # Return the assistant's code response along with contexts
        return {
            "answer": code,
            "contexts": truncated_contexts
        }
    except Exception as e:
        logging.error(f"Error generating CUDA code: {e}")
        return {
            "answer": "Sorry, I couldn't generate CUDA code at this time.",
            "contexts": []
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
    debugging_prompt = f"""You are an assistant that assists with CUDA code debugging.
Provide debugging steps based on CUDA’s debugging best practices, including handling common issues like synchronization, memory access, and kernel crashes.

User Query:
{user_query}

Relevant Documents (Top 3):
""" + "\n---\n".join(truncated_contexts)

    logging.info(f"User message content prepared for CUDA code debugging.")

    # Incorporate conversation history into messages to maintain context
    # Since Flask is stateless, conversation history management is handled via client-side
    conversation_history = request.json.get('conversation', []) if request.is_json else []

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for exchange in conversation_history[-10:]:  # Limit to last 10 exchanges to manage tokens
        messages.append({"role": "user", "content": exchange.get("user", "")})
        messages.append({"role": "assistant", "content": exchange.get("assistant", "")})

    messages.append({"role": "user", "content": debugging_prompt})
    messages.append({"role": "assistant", "content": assistant_message})

    try:
        # Generate debugging response using NeMo Guardrails
        response = rails.generate(messages=messages)
        debugging_steps = response.get('content', '').strip()
        logging.info(f"Generated debugging steps: {debugging_steps[:100]}...")  # Log first 100 characters

        # Ensure the debugging steps do not exceed 400 tokens
        debugging_tokens = len(debugging_steps.split())
        if debugging_tokens > 400:
            debugging_steps = ' '.join(debugging_steps.split()[:400])
            logging.warning("Debugging steps truncated to 400 tokens.")

        # Return the assistant's debugging response along with contexts
        return {
            "answer": debugging_steps,
            "contexts": truncated_contexts
        }
    except Exception as e:
        logging.error(f"Error generating debugging steps: {e}")
        return {
            "answer": "Sorry, I couldn't generate debugging steps at this time.",
            "contexts": []
        }

@app.route('/complete', methods=['POST'])
def complete():
    """
    Endpoint to handle CUDA-related user queries.
    Expects a JSON payload with the user's query under the key 'question'.
    Optionally, can include 'conversation' for maintaining context.
    
    Example Request:
    {
        "question": "What is CUDA?",
        "conversation": [
            {"user": "Previous question", "assistant": "Previous answer"},
            ...
        ]
    }
    
    Expected Response:
    {
        "answer": "Your answer here.",
        "contexts": [
            "Relevant document 1",
            "Relevant document 2",
            "Relevant document 3"
        ]
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload."}), 400

        user_query = data.get('question', '').strip()

        if not user_query:
            return jsonify({"error": "No query provided."}), 400

        logging.info(f"Received query: {user_query}")

        # Detect intent
        detected_intent = detect_user_intent(user_query)

        # Handle intents accordingly
        if detected_intent == 'code_generation':
            result = generate_cuda_code(user_query)
        elif detected_intent == 'code_debugging':
            result = handle_code_debugging(user_query)
        elif detected_intent == 'cuda_explanation':
            result = handle_cuda_explanation(user_query)
        elif detected_intent == 'translation':
            result = handle_general_question(user_query, detected_intent)
        elif detected_intent == 'general_question':
            result = handle_general_question(user_query, detected_intent)
        else:
            # Handle unrecognized intents as general questions
            logging.warning(f"Unrecognized intent '{detected_intent}'. Defaulting to 'general_question'.")
            result = handle_general_question(user_query, 'general_question')

        answer = result.get("answer", "Sorry, I couldn't process your request.")
        contexts = result.get("contexts", [])

        # Return the assistant's response along with contexts
        return jsonify({
            "answer": answer,
            "contexts": contexts
        }), 200

    except Exception as e:
        logging.error(f"Error in /complete endpoint: {e}")
        return jsonify({"error": "Internal Server Error."}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """
    Endpoint to reset the conversation history.
    Since Flask is stateless, this endpoint can be used to clear any server-side session if implemented.
    
    Example Request:
    {
        "reset": 1
    }
    
    Expected Response:
    {
        "reset_response": "success"
    }
    """
    try:
        data = request.get_json()
        reset_flag = data.get('reset', 0)
        if reset_flag != 1:
            return jsonify({"error": "Invalid reset request."}), 400

        # Implement session or state reset logic if server-side sessions are used
        # Since the current implementation is stateless, simply acknowledge the reset
        logging.info("Conversation history reset requested.")
        return jsonify({"reset_response": "success"}), 200

    except Exception as e:
        logging.error(f"Error in /reset endpoint: {e}")
        return jsonify({"error": "Failed to reset conversation history."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
