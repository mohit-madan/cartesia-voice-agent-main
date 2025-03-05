from openai import OpenAI
import asyncio
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

# Load environment variables from .env file
load_dotenv()
client = OpenAI()

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAQ data and build FAISS index
with open('./data/wise_faq_with_embeddings.json', 'r') as f:
    faq_data = json.load(f)

# Extract embeddings and build lookup
embeddings = np.array([doc['embedding'] for doc in faq_data]).astype('float32')
doc_lookup = {i: doc for i, doc in enumerate(faq_data)}

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def get_relevant_docs(query_embedding, k=3, distance_threshold=0.5):
    query_vec = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vec, k)
    # If the best match is too far away, return an empty list
    print(distances[0][0], distance_threshold)
    if distances[0][0] <= distance_threshold:
        return []
    relevant_docs = [doc_lookup[i] for i in indices[0]]
    return relevant_docs

def generate_final_response(query):
    # Embed the query
    query_embedding = embed_query(query)
    
    # Retrieve documents with a confidence check
    relevant_docs = get_relevant_docs(query_embedding, k=3, distance_threshold=0.0)
    print(relevant_docs)
    # If no relevant documents are found, trigger the fallback
    if not relevant_docs:
        fallback_response = "I'm sorry, I couldn't find an answer to your question from our documentation. Let me connect you with a support specialist."
        call_tool(query)  # Trigger a tool call (e.g., escalate, log, or notify)
        return fallback_response
    
    # Build the prompt for the language model
    prompt = build_prompt(query, relevant_docs)
    
    # Generate the answer using the language model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            *create_initial_context(),
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    answer = response.choices[0].message.content
    
    # Check if the answer appears to be unsatisfactory (implement your own heuristic here)
    if not answer or "I am not sure" in answer:
        fallback_response = "I'm sorry, I'm having trouble finding the right answer. I'll connect you with a support specialist for further assistance."
        call_tool(query)
        return fallback_response

    return answer

def call_tool(query):
    # Example: log the query or call an external API to escalate
    # This could be as simple as writing to a log file or triggering an alert.
    print(f"Escalating query: {query}")
    # You could also add code here to send a notification or call another service.

def embed_query(query):
    # Generate an embedding for the incoming query
    query_embedding = model.encode(query)
    return query_embedding


def build_prompt(query, relevant_docs):
    # Extract content from the documents
    context = "\n\n".join([doc.get('content', '') for doc in relevant_docs])
    prompt = f"Answer the following customer support query using the information below:\n\nQuery: {query}\n\nDocumentation:\n{context}"
    return prompt

def create_initial_context():
    return [
        {
            "role": "system",
            "content": """You are a helpful Wise customer support agent. Answer questions using only the context provided. If the needed information isn’t there, kindly let the customer know and offer to connect them with a human agent. Keep your tone friendly, professional, conversational, and concise."""
        }
    ]

async def test_llm_responses():
    """Interactive CLI to test LLM responses directly"""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file:")
        print('OPENAI_API_KEY="your-api-key-here"')
        return

    context = create_initial_context()
    
    print("\nOpenAI Response Tester (Press Ctrl+C to exit)")
    print("--------------------------------------------")
    
    try:
        while True:
            user_input = input("\nYou: ")
            if not user_input.strip():
                continue
            
            # Use generate_final_response which handles RAG and fallbacks
            final_response = generate_final_response(user_input)
            print(f"\nAssistant: {final_response}")
            
            # Only add to context if it's not a fallback response
            if "connect you with a support specialist" not in final_response:
                context.append({"role": "user", "content": user_input})
                context.append({"role": "assistant", "content": final_response})
                
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    asyncio.run(test_llm_responses())