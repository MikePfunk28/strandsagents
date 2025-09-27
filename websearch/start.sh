#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-local}"          # local | docker
SPIDER="${2:-site_sitemap}" # site_sitemap | site_crawl
shift || true; shift || true

# Any extra args you pass go straight to scrapy (e.g., -a base=... -a sitemap=...)
EXTRA_ARGS="$*"

if [[ "$MODE" == "local" ]]; then
  # ensure venv active; if not, try .venv
  if [[ -z "${VIRTUAL_ENV:-}" && -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate || true
  fi

  # install deps if first run
  if [[ -f "pyproject.toml" ]]; then
    python -m pip install -e .
  else
    python -m pip install -r requirements.txt
  fi

  # playwright browser (first run only; safe to repeat)
  python -m playwright install chromium

  echo "[local] running spider: ${SPIDER}"
  exec scrapy crawl "${SPIDER}" ${EXTRA_ARGS}

elif [[ "$MODE" == "docker" ]]; then
  IMAGE="${IMAGE:-webscraper:dev}"

  echo "[docker] building ${IMAGE} ..."
  docker build -t "${IMAGE}" .

  echo "[docker] running spider: ${SPIDER}"
  # pass args to scrapy in container
  exec docker run --rm -it \
    -e PYTHONUNBUFFERED=1 \
    "${IMAGE}" \
    scrapy crawl "${SPIDER}" ${EXTRA_ARGS}
else
  echo "Usage: ./start.sh [local|docker] [site_sitemap|site_crawl] <scrapy-args>"
  echo "Ex:    ./start.sh local site_sitemap -a base=https://example.com -a sitemap=https://example.com/sitemap.xml"
  echo "Ex:    ./start.sh docker site_crawl -a base=https://example.com -a start=/blog/"
  exit 1
fi
