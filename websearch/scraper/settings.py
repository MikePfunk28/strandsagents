BOT_NAME = "webscraper"
SPIDER_MODULES = ["scraper.spiders"]
NEWSPIDER_MODULE = "scraper.spiders"

ROBOTSTXT_OBEY = True  # honor robots.txt (RFC 9309)
CONCURRENT_REQUESTS = 8
DOWNLOAD_TIMEOUT = 25

# Politeness
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
RETRY_TIMES = 2

# Cache (optional)
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600

# Playwright integration
DOWNLOADER_MIDDLEWARES = {
    "scrapy_playwright.middleware.ScrapyPlaywrightDownloaderMiddleware": 543,
}
PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 15000
PLAYWRIGHT_LAUNCH_OPTIONS = {"args": ["--no-sandbox"]}

ITEM_PIPELINES = {
    "scraper.pipelines.EphemeralStorePipeline": 300,
}

# Only render with Playwright when requested via meta={"playwright": True}
