import os
import sys
import pandas as pd
import numpy as np
import yaml
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 加载配置
with open(os.path.join(project_root, 'config.yaml'), 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 创建Dash应用
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# 设置标题
app.title = "市场研究数据分析平台"

# 定义布局
app.layout = html.Div([
    # 导航栏
    html.Nav(
        html.Div([
            html.A("市场研究数据分析平台", className="navbar-brand fw-bold", href="#"),
            html.Div([
                dcc.Input(
                    id="search-input",
                    type="text",
                    placeholder="输入研究关键词...",
                    className="form-control me-2"
                ),
                html.Button("搜索", id="search-button", className="btn btn-outline-primary")
            ], className="d-flex")
        ], className="container-fluid"),
        className="navbar navbar-expand-lg navbar-light bg-light mb-4"
    ),
    
    # 主要内容
    html.Div([
        # 顶部统计卡片
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("总数据量", className="card-title"),
                        html.H3(id="total-count", className="card-text")
                    ], className="card-body")
                ], className="card text-white bg-primary")
            ], className="col-md-3"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("高质量数据", className="card-title"),
                        html.H3(id="high-quality-count", className="card-text")
                    ], className="card-body")
                ], className="card text-white bg-success")
            ], className="col-md-3"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("平均质量分", className="card-title"),
                        html.H3(id="avg-quality", className="card-text")
                    ], className="card-body")
                ], className="card text-white bg-info")
            ], className="col-md-3"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("数据来源数", className="card-title"),
                        html.H3(id="source-count", className="card-text")
                    ], className="card-body")
                ], className="card text-white bg-warning")
            ], className="col-md-3")
        ], className="row mb-4"),
        
        # 图表区域
        html.Div([
            # 左侧图表
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("质量分数分布", className="card-title"),
                        dcc.Graph(id="quality-chart")
                    ], className="card-body")
                ], className="card shadow-sm")
            ], className="col-md-6 mb-4"),
            
            # 右侧图表
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("数据来源分布", className="card-title"),
                        dcc.Graph(id="source-chart")
                    ], className="card-body")
                ], className="card shadow-sm")
            ], className="col-md-6 mb-4")
        ], className="row"),
        
        # 内容长度分布
        html.Div([
            html.Div([
                html.Div([
                    html.H5("内容长度分布", className="card-title"),
                    dcc.Graph(id="length-chart")
                ], className="card-body")
            ], className="card shadow-sm")
        ], className="row mb-4"),
        
        # 数据表格
        html.Div([
            html.Div([
                html.Div([
                    html.H5("原始数据表", className="card-title"),
                    dash_table.DataTable(
                        id="data-table",
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'height': 'auto',
                            'minWidth': '150px',
                            'width': '150px',
                            'maxWidth': '300px',
                            'whiteSpace': 'normal',
                            'textAlign': 'left'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        filter_action="native",
                        sort_action="native"
                    )
                ], className="card-body")
            ], className="card shadow-sm")
        ], className="row mb-4"),
        
        # 隐藏存储
        dcc.Store(id="data-store"),
        
        # 刷新间隔
        dcc.Interval(
            id='interval-component',
            interval=config['dash']['refresh_interval'] * 1000,  # 毫秒
            n_intervals=0
        )
    ], className="container")
])

# 回调函数：加载最新数据
@app.callback(
    Output('data-store', 'data'),
    [Input('interval-component', 'n_intervals'), Input('search-button', 'n_clicks')],
    [State('search-input', 'value')]
)
def load_data(n_intervals, n_clicks, search_value):
    # 获取data/raw目录中的最新JSON文件
    data_dir = os.path.join(project_root, config['data']['raw_path'])
    
    # 确保目录存在
    if not os.path.exists(data_dir):
        return {'data': [], 'timestamp': datetime.now().isoformat()}
    
    data_files = [f for f in os.listdir(data_dir) if f.startswith('market_data_') and f.endswith('.json')]
    
    if not data_files:
        # 返回空数据
        return {'data': [], 'timestamp': datetime.now().isoformat()}
    
    # 找到最新文件
    latest_file = max(data_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
    file_path = os.path.join(data_dir, latest_file)
    
    try:
        # 读取数据
        df = pd.read_json(file_path, lines=True)
        
        # 过滤数据（如果有搜索词）
        if search_value:
            df = df[df['content'].str.contains(search_value, case=False, na=False)]
        
        # 选择要显示的列
        display_columns = ['title', 'source', 'publish_date', 'quality_score', 'crawl_time']
        display_df = df[display_columns].copy() if len(df) > 0 else pd.DataFrame(columns=display_columns)
        
        # 内容长度计算
        if 'content' in df.columns:
            df['content_length'] = df['content'].str.len()
        else:
            df['content_length'] = 0
        
        # 计算统计数据
        stats = {
            'total_count': len(df),
            'high_quality_count': len(df[df['quality_score'] >= 0.6]) if 'quality_score' in df.columns else 0,
            'avg_quality': round(df['quality_score'].mean(), 2) if 'quality_score' in df.columns and len(df) > 0 else 0,
            'source_count': df['source'].nunique() if 'source' in df.columns else 0
        }
        
        # 准备图表数据
        quality_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        quality_labels = ['很低 (0-0.2)', '低 (0.2-0.4)', '中 (0.4-0.6)', '高 (0.6-0.8)', '很高 (0.8-1.0)']
        
        if 'quality_score' in df.columns and len(df) > 0:
            df['quality_group'] = pd.cut(df['quality_score'], bins=quality_bins, labels=quality_labels)
            quality_counts = df['quality_group'].value_counts().to_dict()
        else:
            quality_counts = {label: 0 for label in quality_labels}
        
        # 来源统计
        if 'source' in df.columns and len(df) > 0:
            source_counts = df['source'].value_counts().head(10).to_dict()
        else:
            source_counts = {}
        
        # 内容长度统计
        if 'content_length' in df.columns and len(df) > 0:
            length_stats = {
                'mean': int(df['content_length'].mean()),
                'median': int(df['content_length'].median()),
                'min': int(df['content_length'].min()),
                'max': int(df['content_length'].max()),
                'histogram': df['content_length'].tolist()
            }
        else:
            length_stats = {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'histogram': []
            }
        
        return {
            'data': display_df.to_dict('records'),
            'stats': stats,
            'quality_counts': quality_counts,
            'source_counts': source_counts,
            'length_stats': length_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return {'data': [], 'timestamp': datetime.now().isoformat()}

# 回调函数：更新统计数据
@app.callback(
    [Output('total-count', 'children'),
     Output('high-quality-count', 'children'),
     Output('avg-quality', 'children'),
     Output('source-count', 'children')],
    [Input('data-store', 'data')]
)
def update_stats(data):
    if not data or 'stats' not in data:
        return '0', '0', '0', '0'
    
    stats = data['stats']
    return str(stats['total_count']), str(stats['high_quality_count']), str(stats['avg_quality']), str(stats['source_count'])

# 回调函数：更新数据表格
@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    [Input('data-store', 'data')]
)
def update_table(data):
    if not data or 'data' not in data or not data['data']:
        return [], []
    
    # 准备表格数据
    table_data = data['data']
    columns = [{"name": i, "id": i} for i in table_data[0].keys()]
    
    return table_data, columns

# 回调函数：更新质量分布图表
@app.callback(
    Output('quality-chart', 'figure'),
    [Input('data-store', 'data')]
)
def update_quality_chart(data):
    if not data or 'quality_counts' not in data:
        # 返回空图表
        return px.bar(
            x=[],
            y=[],
            labels={'x': '质量分数', 'y': '数量'},
            title='数据质量分布'
        )
    
    quality_counts = data['quality_counts']
    
    # 创建数据框
    df = pd.DataFrame({
        'quality': list(quality_counts.keys()),
        'count': list(quality_counts.values())
    })
    
    # 创建图表
    fig = px.bar(
        df,
        x='quality',
        y='count',
        color='count',
        color_continuous_scale='Viridis',
        labels={'quality': '质量分数', 'count': '数量'},
        title='数据质量分布'
    )
    
    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    
    return fig

# 回调函数：更新来源分布图表
@app.callback(
    Output('source-chart', 'figure'),
    [Input('data-store', 'data')]
)
def update_source_chart(data):
    if not data or 'source_counts' not in data or not data['source_counts']:
        # 返回空图表
        return px.pie(
            names=[],
            values=[],
            title='数据来源分布'
        )
    
    source_counts = data['source_counts']
    
    # 创建数据框
    df = pd.DataFrame({
        'source': list(source_counts.keys()),
        'count': list(source_counts.values())
    })
    
    # 创建图表
    fig = px.pie(
        df,
        names='source',
        values='count',
        title='数据来源分布',
        hole=0.4
    )
    
    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# 回调函数：更新内容长度分布图
@app.callback(
    Output('length-chart', 'figure'),
    [Input('data-store', 'data')]
)
def update_length_chart(data):
    if not data or 'length_stats' not in data or not data['length_stats']['histogram']:
        # 返回空图表
        return px.histogram(
            x=[],
            labels={'x': '内容长度', 'y': '频率'},
            title='内容长度分布'
        )
    
    length_stats = data['length_stats']
    
    # 创建图表
    fig = px.histogram(
        x=length_stats['histogram'],
        nbins=20,
        labels={'x': '内容长度', 'y': '频率'},
        title='内容长度分布'
    )
    
    # 添加均值线
    fig.add_vline(
        x=length_stats['mean'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"均值: {length_stats['mean']}",
        annotation_position="top right"
    )
    
    # 添加中位数线
    fig.add_vline(
        x=length_stats['median'],
        line_dash="dot",
        line_color="green",
        annotation_text=f"中位数: {length_stats['median']}",
        annotation_position="top left"
    )
    
    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# 启动服务器
if __name__ == '__main__':
    app.run_server(
        host=config['dash']['host'],
        port=config['dash']['port'],
        debug=config['dash']['debug']
    ) 