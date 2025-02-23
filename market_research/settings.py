BOT_NAME = 'market_spider'

SPIDER_MODULES = ['market_research.spiders']
NEWSPIDER_MODULE = 'market_research.spiders'

# 添加Python路径
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 日志设置
LOG_ENABLED = True
LOG_LEVEL = 'DEBUG'
LOG_FILE = 'output/spider.log'
LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
LOG_DATEFORMAT = '%Y-%m-%d %H:%M:%S'

# 爬虫设置
ROBOTSTXT_OBEY = False  # 暂时关闭robots.txt检查以提高数据采集成功率
CONCURRENT_REQUESTS = 4  # 减少并发请求数

# 下载器设置
DOWNLOAD_DELAY = 3  # 下载延迟
DOWNLOAD_TIMEOUT = 30  # 下载超时
RETRY_TIMES = 3  # 重试次数
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 404]  # 重试的HTTP状态码

# 缓存设置
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 86400  # 24小时
HTTPCACHE_DIR = 'data/cache'

# 自动限速
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_DEBUG = True

# 请求头设置
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}
COOKIES_ENABLED = False

# 下载中间件
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
    'scrapy.downloadermiddlewares.robotstxt.RobotsTxtMiddleware': None,
}
DOWNLOAD_DELAY = 3  # 增加下载延迟
RANDOMIZE_DOWNLOAD_DELAY = True  # 随机化延迟
COOKIES_ENABLED = False

# 自动限速设置
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = True

# 输出设置
FEED_EXPORT_ENCODING = 'utf-8'
FEED_FORMAT = 'csv'

# 用户代理设置
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
}

# 重试设置
RETRY_ENABLED = True
RETRY_TIMES = 10  # 增加重试次数
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429, 403, 404]  # 添加更多需要重试的错误码

# 下载超时设置
DOWNLOAD_TIMEOUT = 30

# 缓存设置
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 0
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = []
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# 输出设置
FEEDS = {
    'output/market_data.json': {
        'format': 'jsonlines',
        'encoding': 'utf8',
        'store_empty': False,
        'overwrite': True,
    }
}

# 日志设置
LOG_ENABLED = True
LOG_LEVEL = 'INFO'
LOG_FILE = 'output/spider.log'

# 下载中间件设置
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': 400,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
}

# 用户代理设置
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'

# 超时设置
DOWNLOAD_TIMEOUT = 30

# 质量设置
MIN_CONTENT_LENGTH = 200
MAX_CONTENT_LENGTH = 50000
MIN_QUALITY_SCORE = 0.6
