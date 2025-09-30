#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

def extract_imports_from_file(file_path):
    """Extract all import statements from a Python file."""
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Find all import statements
        # Standard imports: import module
        standard_imports = re.findall(r'^import\s+([^\s]+)', content, re.MULTILINE)

        # From imports: from module import something
        from_imports = re.findall(r'^from\s+([^\s]+)\s+import', content, re.MULTILINE)

        # From imports with dots: from package.module import something
        dotted_imports = re.findall(r'^from\s+([^\s.]+(?:\.[^\s.]+)*)', content, re.MULTILINE)

        # Add all found imports
        for imp in standard_imports:
            # Remove any trailing comments or continuation
            imp = imp.split(',')[0].split(' as ')[0].strip()
            if imp:
                imports.add(imp)

        for imp in from_imports:
            # Remove any trailing comments or continuation
            imp = imp.split(',')[0].split(' as ')[0].strip()
            if imp:
                imports.add(imp)

        for imp in dotted_imports:
            if imp:
                imports.add(imp)

    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)

    return imports

def find_python_files(directory):
    """Find all Python files in directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_imports.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        sys.exit(1)

    print(f"Scanning directory: {directory}")

    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files")

    all_imports = set()

    for file_path in python_files:
        imports = extract_imports_from_file(file_path)
        all_imports.update(imports)
        if imports:
            print(f"{file_path}: {len(imports)} imports")

    # Write all imports to file
    with open('all_imports.txt', 'w') as f:
        for imp in sorted(all_imports):
            f.write(f"{imp}\n")

    print(f"\nTotal unique imports found: {len(all_imports)}")
    print("Imports written to all_imports.txt")

if __name__ == "__main__":
    main()
