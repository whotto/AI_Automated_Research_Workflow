import os
import sys
import yaml
import json
import subprocess
import time
from datetime import datetime
import pandas as pd
import re
from datetime import datetime
import openai
from dotenv import load_dotenv
import logging

# LangChain 相关
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class ResearchWorkflow:
    def __init__(self):
        # 创建必要的目录
        os.makedirs('output', exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # 初始化章节关键词映射
        self.section_keywords = {
            '市场规模': ['市场规模', '市值', '产值', '销售额', '增长率'],
            '市场结构': ['市场结构', '细分市场', '市场份额', '集中度'],
            '产业链': ['产业链', '上游', '下游', '供应链', '价值链'],
            '竞争格局': ['竞争格局', '竞争态势', '市场竞争', '竞争对手'],
            '波特五力': ['竞争者', '供应商', '购买者', '替代品', '进入者'],
            '企业分析': ['公司', '企业', '厂商', '品牌', '商家'],
            '技术趋势': ['技术', '创新', '研发', '专利', '工艺'],
            '市场趋势': ['趋势', '发展', '变化', '前景', '机遇'],
            '政策趋势': ['政策', '法规', '监管', '标准', '规范'],
            '风险': ['风险', '挑战', '问题', '困难', '威胁'],
            '建议': ['建议', '策略', '方案', '对策', '规划']
        }
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
    def _extract_keywords(self, section_title):
        """从章节标题中提取关键词
        
        Args:
            section_title (str): 章节标题
            
        Returns:
            list: 相关的关键词列表
        """
        # 移除章节编号
        title = re.sub(r'^\d+\.\d+\s+', '', section_title)
        
        # 查找最匹配的关键词集
        best_match = None
        max_similarity = 0
        
        for key, keywords in self.section_keywords.items():
            similarity = self._calculate_similarity(title, key)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = keywords
        
        return best_match if best_match else [title]
    
    def _calculate_similarity(self, str1, str2):
        """计算两个字符串的相似度
        
        Args:
            str1 (str): 第一个字符串
            str2 (str): 第二个字符串
            
        Returns:
            float: 相似度得分 (0-1)
        """
        # 使用简单的字符重叠率作为相似度度量
        set1 = set(str1)
        set2 = set(str2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0

    def __init__(self):
        # 创建必要的目录
        os.makedirs('output', exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # 初始化章节关键词映射
        self.section_keywords = {
            '市场规模': ['市场规模', '市值', '产值', '销售额', '增长率'],
            '市场结构': ['市场结构', '细分市场', '市场份额', '集中度'],
            '产业链': ['产业链', '上游', '下游', '供应链', '价值链'],
            '竞争格局': ['竞争格局', '竞争态势', '市场竞争', '竞争对手'],
            '波特五力': ['竞争者', '供应商', '购买者', '替代品', '进入者'],
            '企业分析': ['公司', '企业', '厂商', '品牌', '商家'],
            '技术趋势': ['技术', '创新', '研发', '专利', '工艺'],
            '市场趋势': ['趋势', '发展', '变化', '前景', '机遇'],
            '政策趋势': ['政策', '法规', '监管', '标准', '规范'],
            '风险': ['风险', '挑战', '问题', '困难', '威胁'],
            '建议': ['建议', '策略', '方案', '对策', '规划']
        }
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
            timeout=float(os.getenv('OPENAI_REQUEST_TIMEOUT', 600)),
            max_retries=int(os.getenv('OPENAI_MAX_RETRIES', 10))
        )
    
    def analyze_requirements(self, user_query):
        """分析用户需求并生成调研大纲"""
        logger.info("开始分析需求并生成调研大纲...")
        
        # 定义输出解析器
        response_schemas = [
            ResponseSchema(name="outline", description="调研大纲，包含研究背景、目标、问题维度等"),
            ResponseSchema(name="search_keywords", description="搜索关键词列表，用于数据采集"),
            ResponseSchema(name="report_structure", description="报告结构，包含各章节标题和内容要点")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        # 创建提示模板
        template = """作为一个专业的市场调研分析师，请基于用户的调研需求生成一份完整的调研计划。

用户需求：{query}

请提供以下内容：
1. 一份专业的调研大纲，包含：
   - 研究背景和目标
   - 具体需要调研的问题和维度
   - 数据采集的重点内容
   - 需要采集的信息来源

2. 针对性的搜索关键词列表，确保覆盖：
   - 行业概况相关
   - 市场规模相关
   - 竞争格局相关
   - 技术发展相关
   - 政策环境相关

3. 详细的报告结构，包含：
   - 每个章节的标题
   - 每个章节需要包含的要点
   - 每个章节的数据需求

{format_instructions}

请确保内容专业、全面且具有针对性。"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        
        # 创建语言模型链
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            max_tokens=16384,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        
        try:
            # 执行分析
            result = chain.run(query=user_query)
            
            # 尝试解析返回结果
            try:
                parsed_result = output_parser.parse(result)
            except Exception as parse_error:
                logger.error(f"JSON 解析错误: {parse_error}")
                logger.debug(f"Raw result: {result}")
                
                # 尝试修复 JSON 格式
                fixed_result = result.replace("'", '"')
                fixed_result = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', fixed_result)
                
                try:
                    parsed_result = output_parser.parse(fixed_result)
                except Exception as e:
                    logger.error(f"修复后的 JSON 仍然无法解析: {e}")
                    raise
            
            logger.info("需求分析完成")
            # 确保搜索关键词是列表格式
            search_keywords = parsed_result['search_keywords']
            if isinstance(search_keywords, str):
                search_keywords = [kw.strip() for kw in search_keywords.split(',')]
            elif not isinstance(search_keywords, list):
                search_keywords = [str(search_keywords)]
            
            return (
                parsed_result['outline'],
                search_keywords,
                parsed_result['report_structure']
            )
        except Exception as e:
            logger.error(f"需求分析失败: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"API错误: {e}")
            raise
        
        return outline, keywords
    
    def run_spider(self, keywords):
        """运行爬虫收集数据"""
        logger.info("开始运行爬虫...")
        try:
            # 创建数据目录
            os.makedirs('output', exist_ok=True)
            os.makedirs('data/raw', exist_ok=True)
            os.makedirs('data/processed', exist_ok=True)
            
            # 清理旧的输出文件
            output_file = 'output/market_data.json'
            if os.path.exists(output_file):
                os.remove(output_file)
            
            # 处理关键词
            if isinstance(keywords, str):
                keywords_list = [k.strip() for k in keywords.split(',')]
            else:
                keywords_list = [k.strip() for k in keywords]
            keywords_str = ','.join(keywords_list)
            
            # 加载爬虫设置
            from scrapy.settings import Settings
            from scrapy.crawler import CrawlerProcess
            from market_research.spiders.market_spider import MarketSpider
            
            # 设置爬虫参数
            custom_settings = {
                'CONCURRENT_REQUESTS': 4,
                'DOWNLOAD_DELAY': 3,
                'COOKIES_ENABLED': False,
                'RETRY_ENABLED': True,
                'RETRY_TIMES': 3,
                'RETRY_HTTP_CODES': [500, 502, 503, 504, 522, 524, 408, 429, 403, 404],
                'ROBOTSTXT_OBEY': False,
                'FEEDS': {
                    output_file: {
                        'format': 'jsonlines',
                        'encoding': 'utf8',
                        'overwrite': True
                    }
                },
                'LOG_ENABLED': True,
                'LOG_LEVEL': 'DEBUG',
                'LOG_FILE': 'output/spider.log',
                'DOWNLOAD_TIMEOUT': 30,
                'DOWNLOADER_MIDDLEWARES': {
                    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
                    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
                    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
                    'scrapy.downloadermiddlewares.robotstxt.RobotsTxtMiddleware': None,
                },
                'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'DEFAULT_REQUEST_HEADERS': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                },
                'AUTOTHROTTLE_ENABLED': True,
                'AUTOTHROTTLE_START_DELAY': 5,
                'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
                'AUTOTHROTTLE_MAX_DELAY': 60,
                'AUTOTHROTTLE_DEBUG': True,
                'HTTPCACHE_ENABLED': True,
                'HTTPCACHE_EXPIRATION_SECS': 86400,
                'HTTPCACHE_DIR': 'data/cache'
            }
            
            # 创建爬虫进程
            settings = Settings()
            for key, value in custom_settings.items():
                settings.set(key, value)
            process = CrawlerProcess(settings)
            
            # 运行爬虫
            process.crawl(MarketSpider, keywords=keywords_str)
            process.start()
            
            logger.info("爬虫运行完成")
            
            # 检查输出文件是否存在并且非空
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                logger.error("没有采集到数据")
                return []
            
            # 读取JSON数据
            data = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            logger.error(f"解析JSON数据失败: {str(e)}")
            
            if not data:
                logger.error("数据为空")
                return []
            
            # 保存为CSV格式
            df = pd.DataFrame(data)
            df.to_csv('data/raw/market_data.csv', index=False, encoding='utf-8')
            
            return data
            
        except Exception as e:
            logger.error(f"爬虫运行失败: {str(e)}")
            raise
    
    def process_data(self):
        """数据处理和清洗"""
        logger.info("开始处理数据...")
        df = pd.read_csv('data/raw/market_data.csv')
        
        # 数据清洗
        df = df[df['content'].str.len() > self.config['data']['min_content_length']]
        df['date'] = pd.to_datetime(df['publish_date'])
        df = df.dropna(subset=['date'])
        
        # 特征工程
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['text_length'] = df['content'].apply(len)
        
        # 保存处理后的数据
        df.to_parquet('data/processed/cleaned_data.parquet')
        logger.info("数据处理完成")
        return df
    
    def build_model(self, df):
        """构建预测模型"""
        logger.info("开始构建模型...")
        
        # 准备时间序列数据
        ts_data = df.groupby(['year', 'month'])['text_length'].sum().reset_index()
        
        # 初始化PyCaret
        exp = setup(
            data=ts_data,
            target='text_length',
            session_id=self.config['model']['session_id'],
            log_experiment=True,
            experiment_name=self.config['model']['experiment_name']
        )
        
        # 比较并选择最佳模型
        best_model = compare_models(sort='MAE', n_select=1)
        
        # 保存模型
        save_model(best_model, os.path.join(self.config['model']['save_path'], 'forecast_model'))
        logger.info("模型构建完成")
        return best_model
    
    def start_dashboard(self):
        """启动Dash可视化界面"""
        logger.info("启动可视化界面...")
        subprocess.Popen(["python", "dash_app/app.py"])
    
    def run_pipeline(self, user_query):
        """运行完整的研究工作流"""
        try:
            # 1. 需求分析和大纲生成
            logger.info("开始需求分析...")
            outline, keywords, report_structure = self.analyze_requirements(user_query)
            
            # 2. 爬取数据
            logger.info("开始数据采集...")
            scraped_data = self.run_spider(keywords)
            
            if not scraped_data:
                logger.error("没有采集到数据")
                return {"status": "error", "message": "没有采集到数据"}
            
            # 3. 生成报告
            logger.info("生成研究报告...")
            try:
                report = self.generate_report(user_query, outline, scraped_data)
            except Exception as e:
                logger.error(f"生成报告失败: {e}")
                return {"status": "error", "message": f"生成报告失败: {e}"}
            
            # 4. 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"output/market_research_report_{timestamp}.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"报告已生成并保存到: {report_file}")
            
            return {
                "status": "success",
                "report_file": report_file
            }
            
        except Exception as e:
            logger.error(f"工作流运行失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_report(self, query, outline, data):
        """生成研究报告"""
        # 1. 数据预处理
        processed_data = []
        for item in data:
            if item.get('content') and len(item['content']) > 100:
                # 计算数据质量分数
                quality_score = float(item.get('quality_score', 0))
                if quality_score >= 0.6:  # 只使用高质量数据
                    processed_data.append({
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'source': item.get('source', ''),
                        'url': item.get('url', ''),
                        'quality_score': quality_score,
                        'crawl_time': item.get('crawl_time', '')
                    })
        
        # 2. 按数据来源分类
        source_categories = {
            'research': [
                'chyxx', 'qianzhan', 'iresearch', 'analysys', 'cir',
                'ccid', 'leadleo', 'forward'
            ],
            'industry': [
                'instrument', 'semi', 'dramx', 'cinn'
            ],
            'news': [
                'sohu', 'sina', 'eastmoney', 'yicai', 'caixin', '21jingji'
            ],
            'academic': [
                'cnki', 'scholar'
            ]
        }
        
        categorized_data = {
            'research': [],    # 专业研究报告
            'industry': [],    # 行业资讯
            'news': [],        # 新闻媒体
            'academic': [],    # 学术资料
            'other': []        # 其他来源
        }
        
        for item in processed_data:
            source = item['source']
            category = 'other'
            for cat, sources in source_categories.items():
                if any(s in source for s in sources):
                    category = cat
                    break
            categorized_data[category].append(item)
        
        # 3. 提取数据指标
        import re
        data_metrics = {
            'market_size': [],      # 市场规模
            'growth_rate': [],      # 增长率
            'market_share': [],     # 市场份额
            'investment': [],       # 投资数据
            'forecast': []          # 预测数据
        }
        
        patterns = {
            'market_size': [r'\d+\.?\d*亿', r'\d+\.?\d*万', r'\d+\.?\d*兆'],
            'growth_rate': [r'\d+\.?\d*%', r'CAGR', r'复合增长率'],
            'market_share': [r'市场份额.{0,10}\d+\.?\d*%', r'占据.{0,10}\d+\.?\d*%'],
            'investment': [r'投资.{0,20}\d+\.?\d*(亿|万|兆)'],
            'forecast': [r'\d{4}.{0,10}预计', r'\d{4}.{0,10}展望']
        }
        
        for category, items in categorized_data.items():
            for item in items:
                content = item['content']
                for metric, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            data_metrics[metric].append({
                                'value': match.group(),
                                'source': item['source'],
                                'url': item['url'],
                                'context': content[max(0, match.start()-50):min(len(content), match.end()+50)]
                            })
        
        # 4. 生成报告
        prompt = PromptTemplate(
            input_variables=["query", "outline", "data", "metrics"],
            template="""请基于以下信息生成一份专业的市场调研报告，严格遵循规范格式：

研究需求：{query}

调研大纲：
{outline}

数据来源：
{data}

关键指标：
{metrics}

请生成一份详尽的调研报告，包含以下部分：

1. 摘要（0-1）
- 简要总结报告的主要发现和结论

2. 行业基本面（1-3）
- 行业定位：产品与价值主张
- 商业模式：改造/替代/创新
- 行业周期与成熟度

3. 波特五力分析（3-5）
- 现有竞争者的竞争程度
- 供应商的谈判能力
- 购买者的谈判能力
- 替代品的威胁
- 新进入者的威胁

4. 价值链分析（5-7）
- 主要价值活动分析
  * 研发与技术
  * 采购与供应链
  * 生产与运营
  * 市场与销售
  * 售后与服务
- 价值链环节效率
- 价值链优化空间

5. 策略定位分析（7-9）
- 战略竞争优势（高价值/低成本）
- 竞争策略类型（差异化/成本领先）
- 市场定位与组合

6. 组织能力分析（9-11）
- 战略（Strategy）：战略目标与规划
- 结构（Structure）：组织架构与职能
- 系统（Systems）：运营系统与流程
- 员工（Staff）：人才结构与管理
- 技能（Skills）：核心能力与优势
- 风格（Style）：管理风格与文化
- 共同价值观（Shared Values）

7. 产品组合分析（11-13）
- 明星业务（Stars）
- 现金牛业务（Cash Cows）
- 问题儿业务（Question Marks）
- 瞎狗业务（Dogs）

8. 发展趋势展望（13-15）
- 市场机遇与挑战
- 未来发展预测
- 投资机会分析

9. 哲学思考（15-16）
- 行业发展的核心命题
- 未来发展的关键思考

在每个分析模块中，请回答以下三个核心问题：
1. 这个维度的当前状态如何？（现状）
2. 这个状态背后的深层原因是什么？（原因）
3. 这个状态未来可能如何变化？（趋势）

要求：
1. 报告字数应在5000-15000字之间
2. 使用Markdown格式，注重排版和可读性
3. 引用数据时必须注明来源链接
4. 尽可能使用具体数据，避免模糊的表述
5. 每个结论都应有数据支持
6. 如果数据不足，请注明“数据缺失”或“数据待补充”
7. 在每个章节末尾加入数据来源引用
8. 在每个章节中加入深入的思考和洞见"""
        )
        
        # 数据预处理和分类
        categorized_data = {
            'research_reports': [],
            'industry_news': [],
            'expert_opinions': [],
            'market_data': []
        }
        
        # 分类数据
        for item in processed_data:
            source = item.get('source', '').lower()
            if source in source_categories['research']:
                categorized_data['research_reports'].append(item)
            elif source in source_categories['industry']:
                categorized_data['industry_news'].append(item)
            elif source in source_categories['news']:
                categorized_data['expert_opinions'].append(item)
            else:
                categorized_data['market_data'].append(item)
        
        # 生成数据指标
        data_metrics = {
            'total_sources': len(processed_data),
            'research_report_count': len(categorized_data['research_reports']),
            'industry_news_count': len(categorized_data['industry_news']),
            'expert_opinions_count': len(categorized_data['expert_opinions']),
            'market_data_count': len(categorized_data['market_data']),
            'data_quality_score': sum(d['quality_score'] for d in processed_data) / len(processed_data) if processed_data else 0
        }
        
        # 将报告分成多个部分生成
        sections = [
            ("1. 摘要", ["简要总结报告的主要发现和结论"]),
            ("2. 行业基本面", [
                "行业定位：产品与价值主张",
                "商业模式：改造/替代/创新",
                "行业周期与成熟度"
            ]),
            ("3. 波特五力分析", [
                "现有竞争者的竞争程度",
                "供应商的谈判能力",
                "购买者的谈判能力",
                "替代品的威胁",
                "新进入者的威胁"
            ]),
            ("4. 价值链分析", [
                "主要价值活动分析",
                "价值链环节效率",
                "价值链优化空间"
            ]),
            ("5. 策略定位分析", [
                "战略竞争优势（高价值/低成本）",
                "竞争策略类型（差异化/成本领先）",
                "市场定位与组合"
            ]),
            ("6. 组织能力分析", [
                "战略（Strategy）：战略目标与规划",
                "结构（Structure）：组织架构与职能",
                "系统（Systems）：运营系统与流程",
                "员工（Staff）：人才结构与管理",
                "技能（Skills）：核心能力与优势",
                "风格（Style）：管理风格与文化",
                "共同价值观（Shared Values）"
            ]),
            ("7. 产品组合分析", [
                "明星业务（Stars）",
                "现金牛业务（Cash Cows）",
                "问题儿业务（Question Marks）",
                "瞎狗业务（Dogs）"
            ]),
            ("8. 发展趋势展望", [
                "市场机遇与挑战",
                "未来发展预测",
                "投资机会分析"
            ]),
            ("9. 哲学思考", [
                "行业发展的核心命题",
                "未来发展的关键思考"
            ])
        ]
        
        full_report = []
        for section_name, section_range in sections:
            # 为每个子章节准备数据
            subsection_data = {}
            for subsection in section_range:
                keywords = self._extract_keywords(subsection)
                relevant_data = [d for d in processed_data if any(kw.lower() in d['content'].lower() for kw in keywords)]
                subsection_data[subsection] = relevant_data
            
            # 准备引用追踪器
            citations = []
            
            # 准备消息列表
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的市场调研分析师，擅长生成详尽的市场调研报告。

报告格式要求：
1. 标题格式：
   - 报告标题使用一级标题(#)
   - 主章节使用二级标题(##)
   - 子章节使用三级标题(###)

2. 章节编号：
   - 主章节从1开始，使用阿拉伯数字
   - 子章节使用十进制数字，如 1.1、1.2 等

3. 数据引用：
   - 使用统一的Markdown链接格式：[数据描述](数据来源URL)
   - 引用链接直接插入在相关数据后面

4. 格式要求：
   - 只在主章节之间使用分隔线(---)
   - 使用适当的空行改善可读性
   - 重要内容使用加粗标记
   - 列表使用无序列表(-)

5. 内容要求：
   - 不要重复研究背景和研究目标
   - 不要在章节末尾添加总结提示语
   - 保持客观和严谨的分析态度
   - 避免使用“我们”“本报告”等主观描述

6. 哲学思考章节特别要求：
   - 作为报告的最后一个章节
   - 使用哲学角度升华全文主题
   - 引用经典哲学言论或名句
   - 提炼行业发展的哲学含义
   - 不超过500字"""
                },
                {
                    "role": "user",
                    "content": """请生成市场调研报告，包含以下章节：

1. 市场规模分析
- 提供详细的市场规模数据
- 分析增长率和未来预测
- 细分市场的占比分析
- 区域市场分布情况

2. 竞争分析
- 主要竞争者的市场份额
- 竞争者的产品优势和特点
- 竞争策略分析
- SWOT分析

3. 技术趋势
- 当前主流技术
- 新兴技术发展
- 技术创新方向
- 专利分析

4. 市场机遇与挑战
- 增长驱动因素
- 市场痛点分析
- 潜在风险
- 发展机遇

5. 政策环境
- 相关政策法规
- 行业标准
- 政策影响分析
- 未来政策趋势

6. 建议与展望
- 市场进入策略
- 产品开发建议
- 营销策略建议
- 风险规避建议

要求：
- 每个章节至少包含500字的分析内容
- 必须包含具体的数据支持
- 提供详细的市场分析和预测
- 加入专业的图表描述
- 结合实际案例进行分析"""
                }
            ]
                messages = [{
                    "role": "user",
                    "content": """
3. 技术趋势:
- 当前主流技术
- 新兴技术发展
- 技术创新方向
- 专利分析"""
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": """
4. 市场机遇与挑战:
- 增长驱动因素
- 市场痛点分析"""
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": """
- 潜在风险
- 发展机遇"""
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": """
5. 政策环境:
- 相关政策法规
- 行业标准"""
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": """
- 政策影响分析
- 未来政策趋势"""
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": """
6. 建议与展望:
- 市场进入策略
- 产品开发建议"""
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": """
- 营销策略建议
- 风险规避建议"""
                    }
                ]
                messages = [
                    {
                        "role": "user",
                        "content": """
要求:
- 每个章节至少包含500字的分析内容
- 必须包含具体的数据支持"""
                    }
                ]
- 提供详细的市场分析和预测
- 加入专业的图表描述
- 结合实际案例进行分析"""
                    }
                ]
                },
                {
                    "role": "user",
                    "content": f"""请生成报告的 {section_name} 部分，包含以下子章节：

Research requirements: {query}

Research outline:
{outline}

Data sources:
{json.dumps(categorized_data, ensure_ascii=False)}

Key metrics:
{json.dumps(data_metrics, ensure_ascii=False)}

Requirements:
1. Content should be comprehensive and well-structured
2. Use specific data with direct Markdown links to sources
3. Analysis should address current status, underlying reasons, and future trends
4. Each conclusion must be supported by linked data sources
5. If data is insufficient, indicate '(Data Missing)' after the statement
6. Use proper Markdown formatting with appropriate spacing
7. Use bold text for important points and findings
8. Keep citations inline with the text, not at the end

Please ensure the output is sufficiently detailed."""}
            ]
            
            max_retries = 3
            retry_delay = 60  # 60秒延迟
            
            for attempt in range(max_retries):
                try:
                    # 添加延迟，但第一次尝试不延迟
                    if attempt > 0:
                        time.sleep(retry_delay)
                    
                    section_response = self.client.chat.completions.create(
                        model="gpt-4-0125-preview",  # 使用最新的GPT-4模型
                        messages=section_messages,
                        temperature=0.8,  # 提高创造性
                        max_tokens=4000,  # 调整到模型支持的最大值
                        top_p=0.9,       # 保持输出的多样性
                        presence_penalty=0.1,  # 鼓励模型探索新的话题
                        frequency_penalty=0.1   # 减少重复内容
                    )
                    section_content = section_response.choices[0].message.content.strip()
                    break  # 成功后跳出循环
                    
                except openai.APIError as e:
                    logger.error(f"API错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:  # 最后一次尝试
                        raise  # 重试次数用完，抛出异常
                    continue  # 否则继续重试
            full_report.append(section_content)
        
        # 添加报告标题
        title = f"# {query}行业深度研究报告\n\n作者：市场研究部\n\n日期：{datetime.now().strftime('%Y-%m-%d')}\n\n---\n\n"
        
        # 合并所有部分
        report = title + "\n\n".join(full_report)
        
        return report

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 获取命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python research_workflow.py <市场研究关键词>\n例如: python research_workflow.py 科学仪器市场调研")
        sys.exit(1)
        
    # 获取用户输入的关键词
    keyword = sys.argv[1]
    
    try:
        # 初始化工作流
        workflow = ResearchWorkflow()
        
        # 运行分析流程
        result = workflow.run_pipeline(keyword)
        
        # 输出结果
        if isinstance(result, dict) and result.get("status") == "success":
            print(f"\n研究报告已生成，请查看文件：{result['report_file']}")
        else:
            print("\n错误：生成报告失败")
    except Exception as e:
        print(f"\n错误：{str(e)}")
