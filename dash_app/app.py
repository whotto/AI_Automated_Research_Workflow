import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pycaret.regression import load_model
import yaml
import os

# 加载配置
with open('../config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 初始化Dash应用
app = dash.Dash(__name__)

def load_data():
    """加载处理后的数据"""
    return pd.read_parquet('../data/processed/cleaned_data.parquet')

def generate_predictions(df):
    """生成预测数据"""
    model = load_model(os.path.join('..', config['model']['save_path'], 'forecast_model'))
    
    # 准备未来6个月的日期
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date, periods=6, freq='M')
    
    # 生成预测
    future_data = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month
    })
    predictions = model.predict(future_data)
    
    return future_dates, predictions

def serve_layout():
    """生成应用布局"""
    df = load_data()
    
    # 加载研究计划和标题
    try:
        with open('../data/processed/research_info.json', 'r', encoding='utf-8') as f:
            research_info = json.load(f)
            title = research_info.get('title', '市场研究分析报告')
            plan = research_info.get('plan', '')
    except Exception as e:
        title = '市场研究分析报告'
        plan = ''
    
    # 按数据来源分组统计
    source_stats = df.groupby('source').size()
    company_stats = df[df['source'] == 'made-in-china'].groupby('company').size().sort_values(ascending=False).head(10)
    
    # 创建数据来源分布图
    fig_sources = px.pie(
        values=source_stats.values,
        names=source_stats.index,
        title='数据来源分布',
        hole=0.3
    )
    
    # 创建企业分布图
    fig_companies = px.bar(
        x=company_stats.index,
        y=company_stats.values,
        title='制造商分布排名',
        labels={'x': '企业名称', 'y': '产品数量'}
    )
    
    # 创建时间趋势图
    news_trend = df[df['source'] == 'baidu_news'].groupby('publish_date').size().reset_index()
    news_trend.columns = ['date', 'count']
    
    fig_trend = px.line(
        news_trend,
        x='date',
        y='count',
        title='市场新闻趋势分析',
        labels={'date': '日期', 'count': '新闻数量'}
    )
    
    # 计算关键指标
    total_products = len(df[df['source'] == 'made-in-china'])
    total_news = len(df[df['source'] == 'baidu_news'])
    total_discussions = len(df[df['source'] == 'zhihu'])
    
    return html.Div([
        html.H1(title, className='header'),
        
        # 研究计划展示
        html.Div([
            html.H2("研究计划", className='section-header'),
            html.Pre(plan, className='plan-content')
        ], className='plan-container'),
        
        # 关键指标卡片
        html.Div([
            html.Div([
                html.H4("监控产品数"),
                html.H2(f"{total_products:,}")
            ], className='metric-card'),
            html.Div([
                html.H4("监控新闻数"),
                html.H2(f"{total_news:,}")
            ], className='metric-card'),
            html.Div([
                html.H4("行业讨论量"),
                html.H2(f"{total_discussions:,}")
            ], className='metric-card')
        ], className='metrics-container'),
        
        # 数据来源分布
        html.Div([
            dcc.Graph(figure=fig_sources)
        ], className='chart-container'),
        
        # 制造商分布
        html.Div([
            dcc.Graph(figure=fig_companies)
        ], className='chart-container'),
        
        # 新闻趋势
        html.Div([
            dcc.Graph(figure=fig_trend)
        ], className='chart-container'),
        
        # 图表展示
        html.Div([
            dcc.Graph(figure=fig_trend),
            dcc.Graph(figure=fig_monthly)
        ], className='charts-container'),
        
        # 数据表格
        html.Div([
            html.H3("原始数据预览"),
            dash.dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
        ], className='table-container')
    ])

app.layout = serve_layout

# 添加CSS样式
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>新能源汽车市场研究分析</title>
        {%favicon%}
        {%css%}
        <style>
            .header {
                text-align: center;
                padding: 20px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
            }
            .metrics-container {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }
            .metric-card {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .charts-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                margin: 20px 0;
            }
            .table-container {
                margin: 20px 0;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(
        host=config['dash']['host'],
        port=config['dash']['port'],
        debug=config['dash']['debug']
    )
