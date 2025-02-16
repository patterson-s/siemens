from cryptography.fernet import Fernet
import numpy as np
import base64

class SecureVectorStore:
    def __init__(self, encryption_key):
        self.fernet = Fernet(encryption_key)
        
    def encrypt_vector(self, vector):
        """Encrypt vector while preserving search capability"""
        # Convert vector to bytes
        vector_bytes = np.array(vector).tobytes()
        
        # Encrypt the vector bytes
        encrypted_bytes = self.fernet.encrypt(vector_bytes)
        
        # Encode for storage
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    
    def decrypt_vector(self, encrypted_str):
        """Decrypt vector from storage"""
        # Decode from storage format
        encrypted_bytes = base64.b64decode(encrypted_str.encode('utf-8'))
        
        # Decrypt the vector
        vector_bytes = self.fernet.decrypt(encrypted_bytes)
        
        # Convert back to vector
        return np.frombuffer(vector_bytes, dtype=np.float32) 