from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PlagiarismDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.document_vectors = None

    def add_document(self, text):
        self.documents.append(text)
        self._update_vectors()
        return len(self.documents) - 1

    def _update_vectors(self):
        self.document_vectors = self.vectorizer.fit_transform(self.documents)

    def check_plagiarism(self, text, threshold=0.8):
        if not self.documents:
            raise ValueError("No documents in the database")
        
        new_vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(new_vector, self.document_vectors).flatten()
        most_similar_index = np.argmax(similarities)
        highest_similarity = similarities[most_similar_index]

        return {
            "most_similar_document_id": int(most_similar_index),
            "similarity_score": float(highest_similarity),
            "is_plagiarism": highest_similarity > threshold
        }

    def get_document(self, document_id):
        if document_id < 0 or document_id >= len(self.documents):
            raise ValueError("Document not found")
        return self.documents[document_id]