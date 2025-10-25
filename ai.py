import numpy as np
import faiss
import ollama
from sentence_teansformers import SentenceTransformer

model = SentenceTransformer('granite-embedding:278m')

# So apparently, this is creating the FAISS index... whatever does that mean XD
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim) # L2 Distance for similarity? What the fuck does that mean?
index.add(embeddings)

# Search... apparently...
your_query = input("What is your query?")
query =  model.encode([your_query])
D, I = indec.search(query, k=2) k nearest neighbors
print(I) # Indices of nearest neighbors... interesting

response = ollama.chat(
    model='granite3.1-moe'
    messages=[
        {
        'role': 'user',
        'content': 'What is love? Baby dont hurt me...'
    }
    ]
)