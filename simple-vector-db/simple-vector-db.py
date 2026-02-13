import numpy as np
from sentence_transformers import SentenceTransformer
class SimpleVectorDB:
    def __init__(self):
        self.items = []
        self.vectors = []
        self.embeddings = []

    def add_vector(self, vector, embedding):
        self.vectors.append(vector)
        self.embeddings.append(embedding)

    def add_items(self, item: str, embedding: np.ndarray):
        self.add_vector(item, embedding)
        self.items.append(item)
    def find_similar(self, query_embedding: np.ndarray, top_k: int = 5):
        if not self.embeddings:
            return []

        similarities = [self.cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.items[i], similarities[i]) for i in top_indices]

transform_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str) -> np.ndarray:
    return transform_model.encode(text)
# Example usage:
if __name__ == "__main__":
    db = SimpleVectorDB()
    texts = ["Hello world", "Machine learning is fun", "I love programming", "Python is great for data science", "Artificial intelligence is the future", "I enjoy coding", "Data analysis is important", "Deep learning advances AI", "Natural language processing is fascinating", "Big data drives insights"]

    for text in texts:
        embedding = embed_text(text)
        db.add_items(text, embedding)

    query = "I enjoy coding"
    query_embedding = embed_text(query)
    similar_items = db.find_similar(query_embedding, top_k=2)

    print("Similar items to query:")
    for item, score in similar_items:
        print(f"Item: {item}, Similarity Score: {score:.4f}")
