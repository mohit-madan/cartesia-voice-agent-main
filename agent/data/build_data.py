import asyncio
import aiohttp
import pickle
import uuid
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Import livekit plugins
from livekit.plugins import openai, rag

# Load environment variables (if you store your API key in .env)
load_dotenv()

# Set the embedding dimension (for text-embedding-3-small, 1536 is typical)
embeddings_dimension = 1536

# Asynchronous function to create an embedding for a given input text
async def _create_embedding(input_text: str, http_session: aiohttp.ClientSession):
    try:
        results = await openai.create_embeddings(
            input=[input_text],
            model="text-embedding-3-small", 
            dimensions=embeddings_dimension,
            http_session=http_session,
        )
        # Return the embedding vector from the first (and only) result
        return results[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

async def main() -> None:
    # Load your documents from the JSON file
    try:
        with open('wise_faq_vector_db1.json', 'r') as f:
            vector_db_docs = json.load(f)
    except FileNotFoundError:
        print("Could not find input JSON file")
        return
    except json.JSONDecodeError:
        print("Error parsing JSON file")
        return
    
    # Initialize the Annoy index builder from the livekit rag plugin
    idx_builder = rag.annoy.IndexBuilder(f=embeddings_dimension, metric="angular")
    
    # Dictionary to map a unique document ID (as a string) to its document
    doc_lookup = {}
    
    # Prepare lists to hold tasks and corresponding document IDs
    tasks = []
    doc_ids = []
    
    async with aiohttp.ClientSession() as session:
        # Create asynchronous tasks for each document
        for doc in vector_db_docs:
            # Combine title and content; adjust as needed
            text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
            if not text.strip():  # Skip empty documents
                continue
                
            # Generate a unique identifier for the document
            doc_uuid = str(uuid.uuid4())
            doc_lookup[doc_uuid] = doc
            doc_ids.append(doc_uuid)
            # Schedule the asynchronous embedding call
            tasks.append(asyncio.create_task(_create_embedding(text, session)))
        
        if not tasks:
            print("No valid documents to process")
            return
            
        # Gather embeddings as tasks complete, with a progress bar
        embeddings_results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating embeddings"):
            embedding = await task
            if embedding:  # Only add valid embeddings
                embeddings_results.append(embedding)
        
        if len(embeddings_results) != len(doc_ids):
            print("Warning: Some embeddings failed to generate")
            # Adjust doc_ids to match successful embeddings
            doc_ids = doc_ids[:len(embeddings_results)]
            
        # Add each embedding to the index builder with its corresponding document ID
        for emb, doc_id in zip(embeddings_results, doc_ids):
            idx_builder.add_item(emb, doc_id)
    
    try:
        # Build the Annoy index and save it to disk
        idx_builder.build()
        idx_builder.save("wise_faq_vdb")
        
        # Save the document lookup mapping using pickle for retrieval
        with open("wise_faq_documents1.pkl", "wb") as f:
            pickle.dump(doc_lookup, f)
    except Exception as e:
        print(f"Error saving data: {e}")

if __name__ == "__main__":
    asyncio.run(main())
