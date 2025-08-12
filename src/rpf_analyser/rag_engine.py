from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import faiss
import torch

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.index = None
        self.documents = []
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end > len(text):
                end = len(text)
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    def index_documents(self, documents: List[str]) -> None:
        """Index documents using BERT embeddings and FAISS"""
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc)
            all_chunks.extend(chunks)
        self.documents = all_chunks

        # Generate embeddings
        embeddings = self.model.encode(all_chunks, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def query(self, question: str, k: int = 3) -> List[str]:
        """Query the index for relevant context"""
        if self.index is None:
            raise ValueError("No documents indexed yet")

        # Generate query embedding
        query_embedding = self.model.encode([question], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().astype('float32')

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return relevant document chunks
        return [self.documents[i] for i in indices[0]]

    def get_most_similar(self, text: str, threshold: float = 0.8) -> List[str]:
        """Get most similar documents above similarity threshold"""
        query_results = self.query(text, k=len(self.documents))
        
        # Calculate similarities
        query_embedding = self.model.encode([text], convert_to_tensor=True)
        similarities = []
        
        for doc in query_results:
            doc_embedding = self.model.encode([doc], convert_to_tensor=True)
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding, doc_embedding
            ).item()
            if similarity >= threshold:
                similarities.append((doc, similarity))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities]