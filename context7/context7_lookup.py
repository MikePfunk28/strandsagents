import requests
import json
import base64
import binascii

# get agentcore and strandsagents documentation


def get_agentcore_docs():
    return get_repo_info('https://api.github.com/repos/agentcore/agentcore')


def get_strandsagents_docs():
    return get_repo_info('https://api.github.com/repos/strandsagents/sdk-python')


def get_repo_info(repo_url):
    """Get repository information with error handling"""
    try:
        response = requests.get(repo_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {repo_url}: {e}")
        return None


def get_readme_content(readme_url):
    """Get README content with error handling"""
    try:
        response = requests.get(readme_url, timeout=10)
        response.raise_for_status()
        readme_data = response.json()
        return base64.b64decode(readme_data['content']).decode('utf-8')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching README from {readme_url}: {e}")
        return None
    except (KeyError, binascii.Error) as e:
        print(f"Error decoding README: {e}")
        return None


# search for user input repo
user_repo_name = input("What AWS repo would you like to search?")

# Repository URLs to check
repos = [
    ('strands-agents/sdk-python', 'StrandsAgents SDK'),
    ('strands-agents/samples', 'Strandsagents Samples'),
    ('aws/bedrock-agentcore-sdk-python', 'AgentCore'),
    (f'aws/{user_repo_name}', 'AWS')
]

for repo_path, repo_name in repos:
    print(f"\n=== {repo_name} ===")

    # Get repository info
    repo_data = get_repo_info(f'https://api.github.com/repos/{repo_path}')
    if repo_data:
        print(f'Name: {repo_data.get("name", "N/A")}')
        print(f'Description: {repo_data.get("description", "N/A")}')
        print(f'URL: {repo_data.get("html_url", "N/A")}')
        print(f'Stars: {repo_data.get("stargazers_count", 0)}')
        print(f'Language: {repo_data.get("language", "N/A")}')

        # Get README
        readme_content = get_readme_content(
            f'https://api.github.com/repos/{repo_path}/readme')
        if readme_content:
            print('README Preview:')
            print(readme_content[:500] + '...')
    else:
        print(f'Repository {repo_path} not found or inaccessible')
