import scrapy
from urllib.parse import urljoin
from readability import Document
from lxml import html as lxml_html
from scraper.items import PageItem


class SiteCrawlSpider(scrapy.Spider):
    """
    Crawl from a seed (when no sitemap exists), staying on-domain.
    Run:
      scrapy crawl site_crawl -a base=https://example.com -a start=/ -a allow="^/blog/"
    """
    name = "site_crawl"
    allowed_domains = []

    def __init__(self, base, start="/", allow="^/", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base = base.rstrip("/")
        self.start_path = start
        self.allowed_domains = [self.base.split("://", 1)[1]]
        self.allow_rx = allow

    def start_requests(self):
        yield scrapy.Request(self.base + self.start_path, callback=self.parse)

    def parse(self, response):
        # Extract
        doc = Document(response.text)
        title = doc.short_title()
        content_html = doc.summary()
        text = lxml_html.fromstring(content_html).text_content().strip()
        yield PageItem(url=response.url, title=title, text=text, html=content_html, metadata={})

        # Follow links (stay on-domain & obey robots; Scrapy handles robots)
        for href in response.css("a::attr(href)").getall():
            url = urljoin(response.url, href)
            if url.startswith(self.base):
                yield scrapy.Request(url, callback=self.parse, dont_filter=False)
