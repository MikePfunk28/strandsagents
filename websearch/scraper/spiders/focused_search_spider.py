import re
import json
import time
import scrapy
from urllib.parse import urlparse, urljoin
from readability import Document
from lxml import html as lxml_html
# ... existing imports ...
from scraper.items import PageItem
# ADD:
from scraper.ollama_rerank import rerank as ollama_rerank

# inside class FocusedSearchSpider:


def __init__(self, q, seeds, max_pages=100, top_k=25, allow=None,
             use_ollama="false", ollama_embed_model="nomic-embed-text", *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.query = q.strip()
    # ...
    self.use_ollama = str(use_ollama).lower() in ("1", "true", "yes", "y")
    self.ollama_embed_model = ollama_embed_model
    # ... rest unchanged ...


def closed(self, reason):
    # build docs list for rerank: title+text
    docs = [it for (_s, _u, it) in self.scored]
    if self.use_ollama and docs:
        ranked = ollama_rerank(self.query, docs, model=self.ollama_embed_model)
        top = [(s, d.get("url", ""), d) for (s, d) in ranked[: self.top_k]]
        print("\n=== Focused Search Results (Ollama semantic top_k) ===")
        for i, (s, u, d) in enumerate(top, 1):
            print(f"{i:02d}. sim={s:.4f}  {u}")
        print("======================================================\n")
    else:
        # fallback to keyword score you already had
        self.scored.sort(key=lambda x: x[0], reverse=True)
        top = self.scored[: self.top_k]
        print("\n=== Focused Search Results (keyword top_k) ===")
        for rank, (s, u, it) in enumerate(top, 1):
            print(f"{rank:02d}. score={s:.2f}  {u}")
        print("==============================================\n")

# Tiny BM25-ish scoring without extra deps (ok for small sets)


def score_text(q_terms, text):
    if not text:
        return 0.0
    text_l = text.lower()
    return sum(1.0 for t in q_terms if t in text_l)


class FocusedSearchSpider(scrapy.Spider):
    """
    Focused, query-driven crawl starting from user-provided seeds.
    Usage (PowerShell example):
      scrapy crawl focused_search `
        -a q="bedrock claude 3.7 release notes" `
        -a seeds="https://aws.amazon.com,https://docs.aws.amazon.com" `
        -a max_pages=100 -a top_k=25
    """
    name = "focused_search"
    custom_settings = {
        # keep polite defaults from settings.py (robots, throttle)
    }

    def __init__(self, q, seeds, max_pages=100, top_k=25, allow=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = q.strip()
        self.q_terms = [t.lower() for t in re.findall(
            r"[A-Za-z0-9_+-]+", self.query) if len(t) > 1]
        self.seeds = [s.strip().rstrip("/")
                      for s in seeds.split(",") if s.strip()]
        if not self.seeds:
            raise ValueError(
                "Provide at least one seed domain via -a seeds=...")
        self.max_pages = int(max_pages)
        self.top_k = int(top_k)
        # by default, stay on the provided hosts
        self.allowed_hosts = {urlparse(s).netloc for s in self.seeds}
        self.visited = set()
        self.scored = []  # (score, url, item_json)
        self.allow_rx = re.compile(allow) if allow else None

    def start_requests(self):
        for s in self.seeds:
            yield scrapy.Request(s, callback=self.parse, dont_filter=True)

    def _same_host(self, url):
        return urlparse(url).netloc in self.allowed_hosts

    def parse(self, response):
        if len(self.visited) >= self.max_pages:
            return
        url = response.url
        if url in self.visited:
            return
        self.visited.add(url)

        # Extract readable content
        try:
            doc = Document(response.text)
            title = doc.short_title() or ""
            content_html = doc.summary()
            text = lxml_html.fromstring(content_html).text_content().strip()
        except Exception:
            title, text, content_html = response.xpath(
                "//title/text()").get() or "", "", ""

        # Light relevance scoring
        s = (score_text(self.q_terms, title) * 3.0) + \
            score_text(self.q_terms, text)
        item = PageItem(url=url, title=title, text=text,
                        html=content_html, metadata={"query": self.query})
        self.scored.append((s, url, dict(item)))

        # Follow links on same hosts (optionally filtered by allow regex)
        for href in response.css("a::attr(href)").getall():
            nxt = urljoin(url, href)
            if not nxt.startswith("http"):
                continue
            if not self._same_host(nxt):
                continue
            if self.allow_rx and not self.allow_rx.search(urlparse(nxt).path or "/"):
                continue
            if nxt not in self.visited and len(self.visited) < self.max_pages:
                yield scrapy.Request(nxt, callback=self.parse)

    def closed(self, reason):
        # Sort and keep top_k; write a small report to stdout for convenience
        self.scored.sort(key=lambda x: x[0], reverse=True)
        top = self.scored[: self.top_k]
        print("\n=== Focused Search Results (top_k) ===")
        for rank, (s, u, it) in enumerate(top, 1):
            print(f"{rank:02d}. score={s:.2f}  {u}")
        print("======================================\n")
