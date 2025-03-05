import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI
from livekit.plugins import openai, rag
import pickle
import numpy as np
import aiohttp

# Load environment variables
load_dotenv()

# Load the same data and index that main.py uses
annoy_index = rag.annoy.AnnoyIndex.load('data/wise_faq_vdb')
embeddings_dimension = 1536

with open('./data/wise_faq_documents1.pkl', 'rb') as f:
    faq_data = pickle.load(f)

async def test_rag_enrichment():
    print("\nRAG Testing Interface (Press Ctrl+C to exit)")
    print("--------------------------------------------")
    
    try:
        while True:
            user_input = input("\nYou: ")
            if not user_input.strip():
                continue
                
            print("\nGenerating embedding...")
            # Create an aiohttp session
            async with aiohttp.ClientSession() as session:
                # Pass the session to create_embeddings
                user_embedding = await openai.create_embeddings(
                    input=[user_input],
                    model="text-embedding-3-small",
                    dimensions=embeddings_dimension,
                    http_session=session  # Pass the session here
                )
            
            print("Querying vector database...")
            # Query the annoy index
            result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
            print(f"\nMatch score: {result}")
            
            # Get the matched document
            result_doc = faq_data[result.userdata]
            print(result_doc)
            print("\nMatched content:")
            print("----------------")
            print(result_doc.get('content', ''))
            
            # Generate response using the context
            print("\nGenerating response...")
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful Wise customer support agent. 
                        Answer questions using only the context provided. 
                        If the needed information isn't there, kindly let the customer know 
                        and offer to connect them with a human agent. 
                        Keep your tone friendly, professional, conversational, and concise."""
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{result_doc.get('content', '')}\n\nQuestion: {user_input}"
                    }
                ]
            )
            print("\nAI Response:")
            print("------------")
            print(response.choices[0].message.content)
            print("\n" + "-"*50)
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    asyncio.run(test_rag_enrichment()) 