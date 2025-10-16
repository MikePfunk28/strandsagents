#!/usr/bin/env python3
"""
Check running models on Ollama and MCP servers
"""
import requests
import subprocess
import json
import sys
from typing import Dict, List, Any

def check_ollama_running_models() -> Dict[str, Any]:
    """Check models currently loaded in Ollama memory"""
    try:
        response = requests.get("http://localhost:11434/api/ps", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection failed: {e}"}

def check_ollama_available_models() -> Dict[str, Any]:
    """Check all models available locally in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection failed: {e}"}

def check_ollama_cli() -> str:
    """Check Ollama via CLI commands"""
    try:
        # Check running models
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"CLI Error: {result.stderr}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"CLI not available: {e}"

def check_mcp_servers() -> List[str]:
    """Check for running MCP servers (basic process check)"""
    try:
        # Check for common MCP server processes
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq node.exe"], 
            capture_output=True, text=True, timeout=10
        )
        mcp_processes = []
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'node.exe' in line and ('mcp' in line.lower() or 'context7' in line.lower()):
                    mcp_processes.append(line.strip())
        return mcp_processes
    except Exception as e:
        return [f"Error checking MCP processes: {e}"]

def main():
    print("Checking Models and Servers...")
    print("=" * 50)
    
    # Check Ollama running models
    print("\nOLLAMA RUNNING MODELS:")
    running = check_ollama_running_models()
    if "error" in running:
        print(f"ERROR: {running['error']}")
    else:
        models = running.get("models", [])
        if models:
            for model in models:
                name = model.get("name", "Unknown")
                size_mb = model.get("size", 0) / (1024 * 1024)
                expires = model.get("expires_at", "N/A")
                print(f"  RUNNING: {name} ({size_mb:.1f}MB) - expires: {expires}")
        else:
            print("  No models currently running")
    
    # Check Ollama available models
    print("\nOLLAMA AVAILABLE MODELS:")
    available = check_ollama_available_models()
    if "error" in available:
        print(f"ERROR: {available['error']}")
    else:
        models = available.get("models", [])
        if models:
            for model in models:
                name = model.get("name", "Unknown")
                size_gb = model.get("size", 0) / (1024 * 1024 * 1024)
                family = model.get("details", {}).get("family", "Unknown")
                params = model.get("details", {}).get("parameter_size", "Unknown")
                print(f"  AVAILABLE: {name} ({size_gb:.1f}GB) - {family} - {params}")
        else:
            print("  No models available")
    
    # Check via CLI
    print("\nOLLAMA CLI STATUS:")
    cli_output = check_ollama_cli()
    if "Error" in cli_output or "not available" in cli_output:
        print(f"ERROR: {cli_output}")
    else:
        print("CLI Output:")
        print(cli_output)
    
    # Check MCP servers
    print("\nMCP SERVERS:")
    mcp_procs = check_mcp_servers()
    if mcp_procs and not any("Error" in proc for proc in mcp_procs):
        for proc in mcp_procs:
            print(f"  RUNNING: {proc}")
    else:
        print("  No MCP servers detected")
        if mcp_procs:
            for proc in mcp_procs:
                if "Error" in proc:
                    print(f"  ERROR: {proc}")

if __name__ == "__main__":
    main()