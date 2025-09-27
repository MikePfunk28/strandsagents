import scrapy


class PageItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()
    html = scrapy.Field()
    published = scrapy.Field()
    price = scrapy.Field()
    metadata = scrapy.Field()
