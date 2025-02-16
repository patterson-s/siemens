from cryptography.fernet import Fernet
from flask import current_app
from app.utils.vector_encryption import SecureVectorStore

class SecureDocumentProcessor:
    def __init__(self):
        self.vector_store = SecureVectorStore(current_app.config['ENCRYPTION_KEY'])
        self.content_cipher = Fernet(current_app.config['ENCRYPTION_KEY'])
    
    def process_document(self, document):
        chunks = self.chunk_document(document.content)
        
        secure_chunks = []
        for chunk in chunks:
            # Generate vector embedding
            vector = self.create_embedding(chunk)
            
            # Encrypt the chunk text
            encrypted_text = self.content_cipher.encrypt(chunk.encode())
            
            # Encrypt the vector while preserving search
            encrypted_vector = self.vector_store.encrypt_vector(vector)
            
            # Store optimized search vector separately
            search_vector = self.optimize_for_search(vector)
            
            secure_chunks.append({
                'document_id': document.id,
                'chunk_text': encrypted_text,
                'encrypted_vector': encrypted_vector,
                'search_vector': search_vector
            })
        
        return secure_chunks 