# preview_sql.py
import os

def read_file_with_fallback(path):
    """Try reading a text file using multiple encodings until one works."""
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                return lines, encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Could not decode file with common encodings.")

def preview_file(filename, line_limit=50):
    """Print a preview of a file with line numbers and detected encoding."""
    print(f"\n=== {filename} ===")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    try:
        lines, encoding = read_file_with_fallback(filename)
        print(f"(Detected encoding: {encoding})\n")
        for i, line in enumerate(lines[:line_limit], 1):
            print(f"{i}: {line}", end='')
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    preview_file('readable_sii.sql')
    preview_file('SII Database backup.sql')
