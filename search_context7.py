import requests
import json

# Search for repositories mentioning both context7 and strands
response = requests.get('https://api.github.com/search/repositories?q=context7+strands')
data = json.loads(response.text)
items = data.get('items', [])

print(f"Found {len(items)} repositories:")
for repo in items[:5]:
    print(f"Repository: {repo['full_name']}")
    print(f"Description: {repo['description']}")
    print(f"URL: {repo['html_url']}")
    print("---")

# Also search for context7 specifically
response2 = requests.get('https://api.github.com/search/repositories?q=context7')
data2 = json.loads(response2.text)
items2 = data2.get('items', [])

print(f"\nFound {len(items2)} repositories mentioning context7:")
for repo in items2[:3]:
    print(f"Repository: {repo['full_name']}")
    print(f"Description: {repo['description']}")
    print("---")
