# 智研数析 (AI Market Research Assistant)

**版本号：** 1.2.0

![市场研究助手](assets/images/banner.png)

## 🔍 简介

智研数析是一款强大的市场研究助手，帮助企业、投资者和分析师快速生成专业的行业研究报告。借助 OpenAI 最新的 **gpt-4o-2024-11-20** 模型，智研数析能够收集数据、分析市场趋势并自动生成洞察深刻的研究报告，大幅提升研究效率。

## 🌟 核心功能

- **一键生成完整研究报告**：只需输入行业关键词，即可获得包含市场规模、竞争格局、发展趋势等全方位分析的专业报告
- **精美数据可视化**：自动生成多种图表（折线图、饼图、雷达图、气泡图等），直观展示市场数据
- **专业分析框架**：集成波特五力、价值链分析、BCG矩阵等经典商业分析框架
- **多维度比较表格**：自动创建企业对比表格、五力评分表格和价值链分析表格
- **针对性建议**：为不同利益相关者（企业、投资者、政策制定者）提供差异化的行动建议

## 📊 应用场景

- **市场调研**：快速了解新市场的规模、结构和增长潜力
- **竞争分析**：深入分析竞争格局和主要企业的战略定位
- **投资决策**：为风投、PE和个人投资者提供全面的行业洞察
- **战略规划**：为企业提供市场趋势和机会点分析，支持战略决策
- **学术研究**：协助研究人员收集和分析行业数据，提供可靠的市场信息

## 🔄 创作流程概览

```
需求分析 → 数据采集 → 数据处理 → 内容生成 → 可视化创建 → 整合报告
```

![工作流程图](assets/images/workflow.png)

1. **需求分析**：智能解析您的研究需求，确定关键研究维度和数据点
2. **数据采集**：自动从多个来源收集相关行业数据和市场信息
3. **数据处理**：清洗和结构化收集到的数据，提取关键指标和趋势
4. **内容生成**：利用gpt-4o-2024-11-20模型生成专业、深度的分析内容
5. **可视化创建**：根据数据特点自动选择最合适的图表类型，呈现关键洞察
6. **整合报告**：将所有内容整合为一份结构完整、逻辑清晰的研究报告

## 🛠️ 使用工具

本项目基于以下先进技术构建：

- [LangChain](https://www.langchain.com/) - AI大型语言模型应用框架
- [OpenAI API](https://openai.com/) - 提供gpt-4o-2024-11-20等先进AI模型
- [Pandas](https://pandas.pydata.org/) - 数据分析和处理
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - 数据可视化

## 📘 使用指南

### 准备工作

1. 准备您的OpenAI API密钥
2. 安装Python 3.9或更高版本
3. 安装所需依赖：`pip install -r requirements.txt`
4. 创建`.env`文件并添加您的API密钥：`OPENAI_API_KEY=your_key_here`

### 开始使用

**方法一：命令行运行（最简单）**

```bash
python research_workflow.py "您想研究的行业关键词"

# 示例
python research_workflow.py "电动汽车市场"
```

**方法二：在Python代码中调用**

```python
from research_workflow import ResearchWorkflow

# 创建工作流实例
workflow = ResearchWorkflow()

# 运行研究流程并获取结果
result = workflow.run_pipeline("半导体设备行业分析")

# 查看报告位置
print(f"报告已生成，保存于: {result['report_file']}")
```

### 报告内容示例

生成的报告通常包含以下内容：

- **专业封面页**：带有报告标题、日期和机构标识
- **执行摘要**：关键发现和建议的简明概述
- **市场概况**：市场规模、增长率和主要细分市场
- **竞争格局**：主要参与者分析和市场份额
- **波特五力分析**：行业竞争强度评估
- **价值链分析**：行业价值创造环节分析
- **发展趋势**：关键技术和市场趋势预测
- **战略建议**：针对不同利益相关者的行动建议

## 🚀 未来计划

- 增加多语言报告生成能力
- 扩展至更多垂直行业的专业分析
- 添加实时数据集成和自动更新功能
- 开发基于历史报告的趋势分析功能
- 增强与其他商业智能工具的集成能力

## 📞 联系我们

- **博客**：[天天悦读](https://yuedu.biz)
- **AI 工作流**：[玄清](https://huanwang.org)
- **Email**：[grow8org@gmail.com](mailto:grow8org@gmail.com)
- **GitHub**：[https://github.com/whotto/](https://github.com/whotto/)

## 📜 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
