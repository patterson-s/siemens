import psycopg2
from psycopg2 import sql
import sys

def init_database(database_url: str, schema_file: str):
    print(f"Connecting to database...")
    
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print(f"Reading schema from {schema_file}...")
        with open(schema_file, 'r', encoding='utf-16') as f:
            schema_sql = f.read()
        
        lines = schema_sql.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('--') or line.startswith('\\'):
                continue
            if 'Name:' in line and 'Type:' in line:
                continue
            if line.startswith('ALTER') and 'OWNER TO postgres' in line:
                continue
            clean_lines.append(line)
        
        clean_sql = '\n'.join(clean_lines)
        statements = clean_sql.split(';')
        
        total = len(statements)
        executed = 0
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            
            try:
                executed += 1
                print(f"[{executed}/{total}] Executing: {statement[:80]}...")
                cursor.execute(statement)
                print(f"[{executed}/{total}] Success")
            except psycopg2.Error as e:
                error_msg = str(e).lower()
                if 'already exists' in error_msg:
                    print(f"[{executed}/{total}] Already exists, skipping...")
                elif 'unrecognized configuration parameter' in error_msg:
                    print(f"[{executed}/{total}] Unsupported parameter, skipping...")
                else:
                    print(f"[{executed}/{total}] Error: {e}")
                    print(f"Statement: {statement[:200]}")
        
        cursor.close()
        conn.close()
        
        print("\nDatabase initialization complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python init_database.py <DATABASE_URL>")
        print("Example: python init_database.py 'postgres://user:pass@host:5432/dbname'")
        sys.exit(1)
    
    database_url = sys.argv[1]
    schema_file = 'readable_sii.sql'
    
    init_database(database_url, schema_file)