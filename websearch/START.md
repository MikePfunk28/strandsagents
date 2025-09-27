## Make it executable
chmod +x start.sh

## Local (regular) run

### first time
python -m venv .venv && source .venv/bin/activate
./start.sh local site_sitemap -a base=https://example.com -a sitemap=https://example.com/sitemap.xml

### or crawl without sitemap

./start.sh local site_crawl -a base=https://example.com -a start=/ -a allow="^/blog/"

## Docker run
./start.sh docker site_sitemap -a base=https://example.com -a sitemap=https://example.com/sitemap.xml

# normal install from the current directory
python -m pip install .

# editable (dev) install if your backend supports PEP 660
python -m pip install -e .

## 1) Local mode (your machine’s Python)

Env: a project-local virtual env at ./.venv

Uses: your OS, your filesystem, your network

Good for: fast iteration on spiders/parsers

What happens:

start.sh local … (or start.ps1 local …) will:

create/activate ./.venv if present

pip install -e . (or -r requirements.txt)

python -m playwright install chromium

run scrapy crawl …

Creates new env here? Yes—one venv in the repo at ./.venv (kept out of Docker).

## 2) Docker mode (containerized run)

Env: an isolated container image built from your Dockerfile

Uses: only what’s in the image (clean, reproducible)

Good for: staging/CI, “works on my machine” avoidance

What happens:

start.sh docker … builds the image and runs scrapy crawl … inside the container.

Playwright browser is installed in the image (not on your host).

## Local (“regular”)
```powershell
# one-time venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps from your chosen file
# if pyproject:
python -m pip install -e .
# if requirements:
pip install -r requirements.txt

# Playwright (only if you actually use playwright rendering later)
python -m playwright install chromium

# REAL SEARCH with Ollama semantic rerank over your chosen domains:
.\start.ps1 local focused_search `
  -a q="bedrock claude 3.7 release notes" `
  -a seeds="https://aws.amazon.com,https://docs.aws.amazon.com" `
  -a max_pages=150 -a top_k=25 `
  -a use_ollama=true `
  -a ollama_embed_model="nomic-embed-text"
```

## Docker

```powershell
# same code, inside a container
# (ensure your Dockerfile installs httpx; if using pyproject it will)
.\start.ps1 docker focused_search `
  -a q="keda scale to zero rabbitmq" `
  -a seeds="https://keda.sh,https://kubernetes.io" `
  -a max_pages=120 -a top_k=20 `
  -a use_ollama=true `
  -a ollama_embed_model="nomic-embed-text"

```