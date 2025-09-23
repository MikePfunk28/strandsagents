import requests
import json
import base64

# Get detailed info about the main context7 repository
response = requests.get('https://api.github.com/repos/upstash/context7')
repo_data = json.loads(response.text)

print('Context7 Repository Information:')
print(f'Name: {repo_data["name"]}')
print(f'Full Name: {repo_data["full_name"]}')
print(f'Description: {repo_data["description"]}')
print(f'URL: {repo_data["html_url"]}')
print(f'Stars: {repo_data["stargazers_count"]}')
print(f'Language: {repo_data["language"]}')
print()

# Get the README content
readme_response = requests.get('https://api.github.com/repos/upstash/context7/readme')
if readme_response.status_code == 200:
    readme_data = json.loads(readme_response.text)
    readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
    print('README Preview:')
    print(readme_content[:1000] + '...')
else:
    print(f'Failed to get README: {readme_response.status_code}')
