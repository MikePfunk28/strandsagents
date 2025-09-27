import json
import re
import scrapy
from urllib.parse import urljoin
from lxml import html as lxml_html
from readability import Document
from scraper.items import PageItem


class SiteSitemapSpider(scrapy.Spider):
    """
    Crawl via sitemap (fastest, most complete) and extract content.
    Run:
      scrapy crawl site_sitemap -a base=https://example.com -a sitemap=https://example.com/sitemap.xml
    """
    name = "site_sitemap"
    custom_settings = {"PLAYWRIGHT_PROCESS_REQUEST_HEADERS": None}

    def __init__(self, base, sitemap=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = base.rstrip("/")
        self.sitemap = sitemap or f"{self.base}/sitemap.xml"

    def start_requests(self):
        yield scrapy.Request(self.sitemap, callback=self.parse_sitemap, dont_filter=True)

    def parse_sitemap(self, response):
        # Scrapy has a SitemapSpider, but we stay explicit for clarity.
        for loc in response.xpath("//*[local-name()='loc']/text()").getall():
            url = loc.strip()
            # Render only if needed (set meta flag based on site patterns)
            yield scrapy.Request(url, callback=self.parse_page, meta={"playwright": False})

    def parse_page(self, response):
        url = response.url
        # Extract HTML
        html = response.text

        # Prefer JSON-LD if present
        jsonld = []
        for s in response.xpath("//script[@type='application/ld+json']/text()").getall():
            try:
                jsonld.append(json.loads(s))
            except Exception:
                pass

        # Readability for main content
        doc = Document(html)
        title = doc.short_title()
        content_html = doc.summary()
        text = lxml_html.fromstring(content_html).text_content().strip()

        # Try to pull publish date / price from JSON-LD
        published, price = None, None

        def walk(obj):
            nonlocal published, price
            if isinstance(obj, dict):
                if "datePublished" in obj and not published:
                    published = obj["datePublished"]
                if "price" in obj and not price:
                    price = str(obj["price"])
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
        walk(jsonld)

        yield PageItem(
            url=url, title=title, text=text, html=content_html,
            published=published, price=price, metadata={"jsonld": jsonld}
        )
