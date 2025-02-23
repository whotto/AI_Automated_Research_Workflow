import scrapy
from scrapy.http import Request
from urllib.parse import urlencode
from datetime import datetime
import json
import re

class MarketSpider(scrapy.Spider):
    name = 'market_spider'
    
    def __init__(self, *args, **kwargs):
        super(MarketSpider, self).__init__(*args, **kwargs)
        
        # 设置允许的域名
        self.allowed_domains = [
            # 搜索引擎
            'www.baidu.com',          # 百度搜索
            'news.baidu.com',         # 百度新闻
            'zhidao.baidu.com',       # 百度知道
            'www.bing.com',           # 必应搜索
            'www.google.com',         # 谷歌搜索
            
            # 知识平台
            'www.zhihu.com',          # 知乎
            'zhuanlan.zhihu.com',     # 知乎专栏
            'www.jianshu.com',        # 简书
            'www.csdn.net',           # CSDN
            
            # 专业研究机构
            'www.chyxx.com',          # 产业信息网
            'www.qianzhan.com',       # 前瞻产业研究院
            'report.iresearch.cn',    # 艾瑞咨询
            'www.analysys.cn',        # 易观智库
            'www.cir.cn',             # 中投顶石
            'www.ccidreport.com',     # 赛迈特
            'www.leadleo.com',        # 头豹研究社
            'www.forward.com.cn',     # 前瀬经纪
            
            # 行业门户
            'www.instrument.com.cn',  # 仪器信息网
            'www.instrument.org.cn',  # 中国仪器仪表行业协会
            'www.cinn.cn',            # 中国仪器仪表网
            'www.semi.org.cn',        # SEMI中国
            'www.dramx.com',          # 存储器行业网
            'www.semiinsights.com',   # 半导体行业观察
            
            # 财经媒体
            'www.sohu.com',           # 搜狐财经
            'finance.sina.com.cn',    # 新浪财经
            'www.eastmoney.com',      # 东方财富
            'www.yicai.com',          # 第一财经
            'www.caixin.com',         # 财新网
            'www.21jingji.com',       # 21世纪经济网
            
            # 企业信息
            'www.tianyancha.com',     # 天眼查
            'www.qcc.com',            # 企查查
            'www.qichacha.com',       # 企查查
            
            # 学术资源
            'www.cnki.net',           # 中国知网
            'xueshu.baidu.com',       # 百度学术
            'scholar.google.com'       # 谷歌学术
        ]
        
        # 从参数获取关键词
        self.keywords = kwargs.get('keywords', '').split(',') if kwargs.get('keywords') else []
        if not self.keywords:
            raise ValueError("必须提供搜索关键词")
            
        # 搜索源配置
        self.search_sources = {
            'baidu': {
                'url': 'https://www.baidu.com/s',
                'params': lambda kw: {'wd': kw + ' 市场规模 行业分析'}
            },
            'news': {
                'url': 'https://news.baidu.com/ns',
                'params': lambda kw: {'word': kw + ' 市场分析', 'tn': 'news'}
            },
            'zhihu': {
                'url': 'https://www.zhihu.com/search',
                'params': lambda kw: {'type': 'content', 'q': kw + ' 行业分析'}
            },
            'chyxx': {
                'url': 'https://www.chyxx.com/search/',
                'params': lambda kw: {'key': kw + ' 行业报告'}
            },
            'qianzhan': {
                'url': 'https://www.qianzhan.com/search/all/',
                'params': lambda kw: {'q': kw + ' 行业分析'}
            },
            'iresearch': {
                'url': 'https://report.iresearch.cn/search/',
                'params': lambda kw: {'keyword': kw + ' 行业报告'}
            },
            'analysys': {
                'url': 'https://www.analysys.cn/article/search',
                'params': lambda kw: {'keyword': kw}
            },
            'cir': {
                'url': 'https://www.cir.cn/search.aspx',
                'params': lambda kw: {'keyword': kw + ' 行业研究'}
            },
            'ccid': {
                'url': 'https://www.ccidreport.com/report/search.html',
                'params': lambda kw: {'keyword': kw}
            },
            'leadleo': {
                'url': 'https://www.leadleo.com/search',
                'params': lambda kw: {'keyword': kw + ' 行业报告'}
            },
            'forward': {
                'url': 'https://www.forward.com.cn/search',
                'params': lambda kw: {'keyword': kw}
            },
            'instrument': {
                'url': 'https://www.instrument.com.cn/lib/search/',
                'params': lambda kw: {'kw': kw + ' 行业分析'}
            },
            'semi': {
                'url': 'https://www.semi.org.cn/search/',
                'params': lambda kw: {'keyword': kw}
            },
            'dramx': {
                'url': 'https://www.dramx.com/Search/',
                'params': lambda kw: {'keyword': kw}
            },
            'caixin': {
                'url': 'https://search.caixin.com/search/',
                'params': lambda kw: {'keyword': kw + ' 行业分析'}
            }
        }
    
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 16,      # 增加并发请求数
        'DOWNLOAD_DELAY': 1,           # 减少下载延迟
        'COOKIES_ENABLED': False,
        'RETRY_TIMES': 5,              # 增加重试次数
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],  # 添加429状态码（太多请求）
        'DOWNLOAD_TIMEOUT': 30,        # 设置下载超时
        'REDIRECT_ENABLED': True,      # 允许重定向
        'HTTPERROR_ALLOWED_CODES': [404, 403],  # 允许的HTTP错误码
        
        # 随机User-Agent
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'SPIDER_MODULES': ['market_research'],
        'NEWSPIDER_MODULE': 'market_research',
        
        # 启用自动限速
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 5,
        'AUTOTHROTTLE_MAX_DELAY': 60,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 4.0,
        
        # 缓存设置
        'HTTPCACHE_ENABLED': True,
        'HTTPCACHE_EXPIRATION_SECS': 86400,  # 24小时
        'HTTPCACHE_DIR': 'data/cache',
        
        # 输出设置
        'FEEDS': {
            'data/raw/market_data_%(time)s.json': {
                'format': 'json',
                'encoding': 'utf8',
                'indent': 2,
                'store_empty': False,
                'fields': ['title', 'content', 'url', 'source', 'keyword', 'crawl_time', 'quality_score']
            }
        },
        
        # 自定义设置
        'MAX_DEPTH': 3,               # 增加最大爬取深度
        'MIN_CONTENT_LENGTH': 500,    # 最小内容长度（字符数）
        'MAX_CONTENT_LENGTH': 50000,  # 最大内容长度（字符数）
        'MIN_QUALITY_SCORE': 0.6      # 最小质量分数（0-1）
    }

    def start_requests(self):
        """生成初始请求"""
        self.logger.info(f"开始爬取，关键词: {self.keywords}")
        for keyword in self.keywords:
            # 对每个搜索源生成请求
            for source, config in self.search_sources.items():
                try:
                    params = config['params'](keyword)
                    url = f"{config['url']}?{urlencode(params)}"
                    self.logger.debug(f"生成请求: {url}")
                    
                    yield Request(
                        url=url,
                        callback=self.parse_search_results,
                        meta={
                            'keyword': keyword,
                            'source': source,
                            'depth': 0
                        },
                        dont_filter=True,
                        errback=self.errback_httpbin,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                        }
                    )
                except Exception as e:
                    self.logger.error(f"生成请求失败: {e}")

    def parse_search_results(self, response):
        """解析搜索结果页面"""
        source = response.meta['source']
        keyword = response.meta['keyword']
        depth = response.meta['depth']
        
        # 根据不同来源选择不同的选择器
        selectors = {
            'baidu': {
                'results': '.result.c-container',
                'title': 'h3 a::text',
                'link': 'h3 a::attr(href)',
                'snippet': '.content::text'
            },
            'news': {
                'results': '.result',
                'title': 'h3 a::text',
                'link': 'h3 a::attr(href)',
                'snippet': '.c-summary::text'
            },
            'zhihu': {
                'results': '.SearchResult-Card',
                'title': '.ContentItem-title::text',
                'link': '.ContentItem-title a::attr(href)',
                'snippet': '.SearchItem-excerpt::text'
            },
            'chyxx': {
                'results': '.search-list li',
                'title': 'a::text',
                'link': 'a::attr(href)',
                'snippet': '.desc::text'
            },
            'qianzhan': {
                'results': '.search-res-list li',
                'title': '.title a::text',
                'link': '.title a::attr(href)',
                'snippet': '.summary::text'
            },
            'iresearch': {
                'results': '.report-item',
                'title': '.report-title::text',
                'link': '.report-title a::attr(href)',
                'snippet': '.report-desc::text'
            },
            'analysys': {
                'results': '.article-item',
                'title': '.article-title a::text',
                'link': '.article-title a::attr(href)',
                'snippet': '.article-desc::text'
            },
            'cir': {
                'results': '.search-result-item',
                'title': '.title a::text',
                'link': '.title a::attr(href)',
                'snippet': '.summary::text'
            },
            'ccid': {
                'results': '.report-list li',
                'title': '.report-name::text',
                'link': 'a::attr(href)',
                'snippet': '.report-desc::text'
            },
            'leadleo': {
                'results': '.search-item',
                'title': '.title::text',
                'link': 'a::attr(href)',
                'snippet': '.desc::text'
            },
            'instrument': {
                'results': '.search-result-item',
                'title': '.title a::text',
                'link': '.title a::attr(href)',
                'snippet': '.summary::text'
            },
            'semi': {
                'results': '.news-item',
                'title': '.news-title::text',
                'link': 'a::attr(href)',
                'snippet': '.news-summary::text'
            },
            'dramx': {
                'results': '.article-item',
                'title': '.article-title::text',
                'link': '.article-title a::attr(href)',
                'snippet': '.article-summary::text'
            },
            'caixin': {
                'results': '.search-result-item',
                'title': '.article-title::text',
                'link': '.article-title a::attr(href)',
                'snippet': '.article-summary::text'
            }
        }
        
        selector = selectors.get(source, selectors['baidu'])
        results = response.css(selector['results'])
        
        for result in results:
            title = result.css(selector['title']).get('')
            url = result.css(selector['link']).get('')
            content = result.css(selector['snippet']).get('')
            
            if title and url:
                yield {
                    'title': title.strip(),
                    'content': content.strip() if content else '',
                    'url': url,
                    'source': source,
                    'keyword': keyword,
                    'crawl_time': datetime.now().isoformat()
                }
        
        for result in results:
            url = result.css(selector['link']).get()
            if not url:
                continue
                
            # 对每个搜索结果进行深度爬取
            if depth < 2:  # 限制爬取深度
                yield Request(
                    url=url,
                    callback=self.parse_content,
                    meta={
                        'keyword': keyword,
                        'source': source,
                        'depth': depth + 1,
                        'title': result.css(selector['title']).get('').strip(),
                        'snippet': result.css(selector['snippet']).get('').strip()
                    }
                )
    
    def calculate_quality_score(self, title, content, url):
        """计算内容质量分数"""
        score = 0.0
        
        # 检查内容长度
        content_length = len(content)
        if content_length < self.settings['MIN_CONTENT_LENGTH']:
            return 0.0
        if content_length > self.settings['MAX_CONTENT_LENGTH']:
            content_length = self.settings['MAX_CONTENT_LENGTH']
        
        # 1. 内容长度分数 (0-0.3)
        length_score = min(0.3, content_length / 5000 * 0.3)
        score += length_score
        
        # 2. 关键词匹配分数 (0-0.2)
        keywords = [
            '市场规模', '市场份额', '市场分析', '行业趋势',
            '竞争格局', '企业分析', '技术趋势', '发展前景',
            '数据', '统计', '调研', '报告', '分析', '研究'
        ]
        keyword_count = sum(1 for kw in keywords if kw in content)
        keyword_score = min(0.2, keyword_count / len(keywords) * 0.2)
        score += keyword_score
        
        # 3. 数据指标分数 (0-0.2)
        data_patterns = [
            r'\d+\.?\d*%',  # 百分比
            r'\d+\.?\d*亿',  # 亿元
            r'\d+\.?\d*万',  # 万元
            r'\d{4}年',  # 年份
            r'CAGR'  # 复合增长率
        ]
        import re
        data_count = sum(1 for pattern in data_patterns if re.search(pattern, content))
        data_score = min(0.2, data_count / len(data_patterns) * 0.2)
        score += data_score
        
        # 4. 来源可信度分数 (0-0.2)
        trusted_domains = [
            'chyxx.com', 'qianzhan.com', 'instrument.com.cn', 'cinn.cn',
            'cnki.net', 'gov.cn', 'org.cn', 'edu.cn'
        ]
        source_score = 0.2 if any(domain in url for domain in trusted_domains) else 0.1
        score += source_score
        
        # 5. 时效性分数 (0-0.1)
        time_patterns = [str(year) for year in range(2023, 2026)]
        time_score = 0.1 if any(year in content for year in time_patterns) else 0.05
        score += time_score
        
        return round(score, 2)
    
    def extract_content(self, response):
        """提取页面内容"""
        # 1. 尝试提取文章正文
        content_selectors = [
            'article p::text',  # 标准文章格式
            '.article-content p::text',  # 常见文章内容类
            '.post-content p::text',  # 博客文章
            '.entry-content p::text',  # WordPress类
            '.content p::text',  # 通用内容类
            '.main-content p::text',  # 主要内容区
            '#article-content p::text',  # 文章ID
            '.detail-content p::text',  # 详情内容
            '.news-content p::text',  # 新闻内容
        ]
        
        content = ''
        for selector in content_selectors:
            paragraphs = response.css(selector).getall()
            if paragraphs:
                content = ' '.join([p.strip() for p in paragraphs if len(p.strip()) > 10])
                if len(content) > 100:  # 如果内容足够长，就使用这个结果
                    break
        
        # 2. 如果没有找到合适的内容，尝试使用摘要
        if not content and response.meta.get('snippet'):
            content = response.meta['snippet']
        
        return content
    
    def parse_content(self, response):
        """解析具体内容页面"""
        try:
            # 提取标题和内容
            title = response.meta.get('title', '')
            content = self.extract_content(response)
            
            # 如果没有提取到内容，使用摘要
            if not content and response.meta.get('snippet'):
                content = response.meta['snippet']
            
            # 过滤无效内容
            if not content or len(content) < 200:
                return
            
            # 计算质量分数
            quality_score = self.calculate_quality_score(title, content, response.url)
            
            # 如果质量分数足够高，才保存数据
            if quality_score >= 0.6:
                data = {
                    'title': title,
                    'content': content,
                    'url': response.url,
                    'source': response.meta['source'],
                    'keyword': response.meta['keyword'],
                    'crawl_time': datetime.now().isoformat(),
                    'quality_score': quality_score
                }
                
                # 尝试写入数据
                try:
                    with open('output/market_data.json', 'a', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                        f.write('\n')
                except Exception as e:
                    self.logger.error(f'写入数据失败: {str(e)}')
                
                yield data
        
        except Exception as e:
            self.logger.error(f'解析内容页面失败: {str(e)}')
    
    def errback_httpbin(self, failure):
        """错误处理函数"""
        self.logger.error(f'请求失败: {str(failure.value)}')

    def parse_detail(self, response):
        item = response.meta['item']
        item['content'] = ' '.join(response.css('article p::text').getall())
        yield item

    def handle_error(self, failure):
        """错误处理"""
        self.logger.error(f'Request failed: {failure.request.url}')
        yield {
            'error': str(failure.value),
            'url': failure.request.url,
            'keyword': failure.request.meta.get('keyword'),
            'source': failure.request.meta.get('source'),
            'crawl_time': datetime.now().isoformat()
        }
