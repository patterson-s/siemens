from cryptography.fernet import Fernet
import os

def generate_encryption_key():
    # Generate a new key
    key = Fernet.generate_key().decode()
    
    # Print the key
    print("\nGenerated Encryption Key:")
    print("-------------------------")
    print(key)
    print("-------------------------")
    
    # Read existing .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    existing_content = ""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            existing_content = f.read()
    
    # Check if ENCRYPTION_KEY already exists
    if 'ENCRYPTION_KEY=' in existing_content:
        print("\nWarning: ENCRYPTION_KEY already exists in .env file!")
        print("Replacing it could make existing encrypted data unreadable.")
        response = input("Do you want to replace it? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
    
    # Update or append the key to .env
    if existing_content and not existing_content.endswith('\n'):
        existing_content += '\n'
    
    with open(env_path, 'w') as f:
        f.write(existing_content)
        if 'ENCRYPTION_KEY=' not in existing_content:
            f.write(f'ENCRYPTION_KEY={key}\n')
        else:
            # Replace existing key
            lines = existing_content.splitlines()
            with open(env_path, 'w') as f:
                for line in lines:
                    if line.startswith('ENCRYPTION_KEY='):
                        f.write(f'ENCRYPTION_KEY={key}\n')
                    else:
                        f.write(f'{line}\n')
    
    print("\nKey has been saved to .env file")
    print("Make sure to backup this key securely!")
    print("If you lose it, encrypted data cannot be recovered.")

if __name__ == '__main__':
    generate_encryption_key() 