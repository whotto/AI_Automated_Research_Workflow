# Scrapy settings for market_research project

BOT_NAME = 'market_research'

SPIDER_MODULES = ['market_research.spiders']
NEWSPIDER_MODULE = 'market_research.spiders'

# 爬取规则设置
ROBOTSTXT_OBEY = False
CONCURRENT_REQUESTS = 8
DOWNLOAD_DELAY = 1
COOKIES_ENABLED = False

# 设置请求超时
DOWNLOAD_TIMEOUT = 15

# 下载中间件
DOWNLOADER_MIDDLEWARES = {
   'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
   'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
   'scrapy.downloadermiddlewares.retry.RetryMiddleware': 500,
}

# 设置重试
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403]

# 默认请求头
DEFAULT_REQUEST_HEADERS = {
   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
   'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

# 启用自动节流
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# 日志设置
LOG_LEVEL = 'INFO'

# 设置输出格式
FEED_EXPORT_ENCODING = 'utf-8'
FEED_FORMAT = 'jsonlines'
FEED_EXPORT_INDENT = None

# 开启收集统计信息
STATS_CLASS = 'scrapy.statscollectors.MemoryStatsCollector'

# 设置文件管道
ITEM_PIPELINES = {
    'scrapy.pipelines.files.FilesPipeline': 1,
} 