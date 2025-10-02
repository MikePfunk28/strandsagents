#!/usr/bin/env python3
import subprocess
import sys
import json

def clean_imports(imports_file):
    """Clean up imports by removing standard library modules."""
    # Standard library modules that shouldn't be in requirements.txt
    stdlib_modules = {
        'importlib.util', 'urllib.parse', 'fmt'
    }

    cleaned_imports = []
    with open(imports_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line not in stdlib_modules:
                cleaned_imports.append(line)

    return cleaned_imports

def resolve_package_name(module_name):
    """Use context7 MCP to resolve module name to package name."""
    try:
        # Use context7 MCP server to get package information
        result = subprocess.run([
            'node', 'C:/Users/mikep/mcp-server/context7-mcp/build/index.js'
        ], input=f'{{"module": "{module_name}"}}',
           text=True, capture_output=True, timeout=30)

        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                if 'package' in response:
                    return response['package']
            except json.JSONDecodeError:
                pass

        # Fallback: try to guess the package name from the module name
        return module_name

    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # If context7 fails, return the module name as-is
        return module_name

def main():
    # Clean the imports first
    filtered_imports = clean_imports('filtered_imports.txt')
    print(f"Cleaned imports: {len(filtered_imports)}")

    # Resolve package names using context7 MCP
    package_requirements = {}
    for module in filtered_imports:
        print(f"Resolving {module}...")
        package = resolve_package_name(module)
        if package != module:  # If context7 gave us a different package name
            package_requirements[package] = module
        else:
            package_requirements[module] = module

    # Write requirements.txt
    with open('requirements_comprehensive.txt', 'w') as f:
        f.write("# Comprehensive requirements.txt generated from all imports in the project\n")
        f.write("# Generated using context7 MCP server for package resolution\n\n")

        for package, module in sorted(package_requirements.items()):
            f.write(f"{package}\n")

    print(f"\nGenerated requirements.txt with {len(package_requirements)} packages")
    print("Requirements written to requirements_comprehensive.txt")

    # Also print the results
    print("\nPackage requirements:")
    for package, module in sorted(package_requirements.items()):
        if package != module:
            print(f"  {module} -> {package}")
        else:
            print(f"  {package}")

if __name__ == "__main__":
    main()
