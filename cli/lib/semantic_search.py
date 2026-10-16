from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

class SemanticSearch():
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Text must not be empty or only whitespace")
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self,documents):
        self.documents = documents
        movies = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")
        # self.embeddings = self.model.encode(movies,show_progress_bar=True)
        # np.save("./cache/movie_embeddings.npy",self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        if os.path.exists("./cache/movie_embeddings.npy"):
            self.embeddings = np.load("./cache/movie_embeddings.npy")
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        scores = []
        for idx,doc in enumerate(self.documents):
            scores.append((cosine_similarity(embedding,self.embeddings[idx]),doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        result = []
        for score in scores[:limit]:
            result.append({"score":score[0],"title":score[1].get("title"),"description":score[1].get("description")})
        return result
        
        

def verify_model():
    search = SemanticSearch()
    MODEL = search.model
    print(f"Model loaded: {MODEL}")
    print("Max sequence length: {MAX_LENGTH}",MODEL.max_seq_length)

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def getData():
    with open("./data/movies.json", "r") as file:
        data = json.load(file)
    data = data["movies"]
    return data

def verify_embeddings():
    search = SemanticSearch()
    data = getData()
    embeddings = search.load_or_create_embeddings(data)
    print(f"Number of docs:   {len(data)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)