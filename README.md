# AI 研究工作流

基于 LangChain + Scrapy + Pandas + PyCaret + Dash 的智能研究工作流系统，用于自动化市场研究和数据分析。

## 功能特点

- 智能需求分析：使用 LangChain 将用户需求转换为结构化研究计划
- 自动数据采集：基于 Scrapy 的智能爬虫系统
- 数据处理分析：使用 Pandas 进行数据清洗和特征工程
- 预测建模：基于 PyCaret 的自动化机器学习建模
- 可视化展示：使用 Dash 构建交互式数据可视化界面

## 项目结构

```
/research_project
├── config.yaml            # 配置文件
├── requirements.txt       # 项目依赖
├── research_workflow.py   # 主程序
├── spiders/              # Scrapy爬虫目录
│   └── market_spider.py
├── data/                 # 数据存储
│   ├── raw/
│   └── processed/
└── dash_app/            # 可视化应用
    └── app.py
```

## 环境配置

1. 创建虚拟环境：
```bash
conda create -n research_ai python=3.9
conda activate research_ai
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并设置以下变量：
```
OPENAI_API_KEY=your_api_key_here
```

## 使用方法

1. 启动研究工作流：
```python
from research_workflow import ResearchWorkflow

workflow = ResearchWorkflow()
result = workflow.run_pipeline("分析2023年北京新能源汽车市场趋势")
```

2. 访问可视化界面：
打开浏览器访问 `http://localhost:8050`

## 自定义配置

编辑 `config.yaml` 文件以修改：
- API 配置
- 爬虫参数
- 数据处理规则
- 模型训练参数
- 可视化设置

## 注意事项

1. 确保已安装所有必要的系统依赖
2. 遵守目标网站的爬虫规则和速率限制
3. 定期备份重要数据
4. 监控系统资源使用情况

## 开发计划

- [ ] 添加更多数据源支持
- [ ] 优化模型训练流程
- [ ] 增强可视化功能
- [ ] 添加自动化测试
- [ ] 改进错误处理机制

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。
