# Using a library like sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import numpy as np
import json
import numpy as np
import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Load vector database documents
with open('wise_faq_vector_db.json', 'r') as f:
    vector_db_docs = json.load(f)

# Create embeddings

for doc in tqdm(vector_db_docs, desc="Creating embeddings"):
    # You might want to split content into chunks if it's too long
    embedding = model.encode(doc['content'])
    doc['embedding'] = embedding.tolist()

# Save documents with embeddings
with open('wise_faq_with_embeddings.json', 'w') as f:
    json.dump(vector_db_docs, f, indent=2)


embeddings = np.array([doc['embedding'] for doc in vector_db_docs]).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "wise_faq.index")

# Save documents without embeddings for retrieval
doc_lookup = [{k: v for k, v in doc.items() if k != 'embedding'} for doc in vector_db_docs]
with open('wise_faq_documents.json', 'w') as f:
    json.dump(doc_lookup, f, indent=2)