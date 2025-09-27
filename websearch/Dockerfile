# dockerfile for websearch agent
# uses scrapy with scrapy-playwright to do headless browser scraping
FROM python:3.12-slim

# system deps needed by Playwright headless Chromium
RUN apt-get update && apt-get install -y \
    libasound2 libatk-bridge2.0-0 libdrm2 libgtk-3-0 libnss3 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

RUN python -m playwright install --with-deps chromium

COPY scrapy.cfg ./
COPY scraper ./scraper
CMD ["scrapy", "list"]