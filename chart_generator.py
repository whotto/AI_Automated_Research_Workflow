import os
import sys
import re
import json
import logging
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('ggplot')
except Exception as e:
    logger.warning(f"设置中文字体支持失败: {e}")

class ChartGenerator:
    """图表生成器，用于自动生成各种类型的图表"""
    
    def __init__(self, output_dir='output/charts'):
        """初始化图表生成器
        
        Args:
            output_dir (str): 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_chart(self, data, chart_type, title, x_label=None, y_label=None, section_name=None):
        """生成图表并保存为图片
        
        Args:
            data (dict): 图表数据，包含x和y数据
            chart_type (str): 图表类型，如'bar', 'line', 'pie', 'scatter'
            title (str): 图表标题
            x_label (str, optional): x轴标签
            y_label (str, optional): y轴标签
            section_name (str, optional): 图表所属章节名称，用于文件命名
            
        Returns:
            str: 图表文件的路径（相对于报告）
        """
        logger.info(f"生成{chart_type}图表: {title}")
        
        # 生成唯一的文件名
        chart_id = str(uuid.uuid4())[:8]
        if section_name:
            # 从section_name提取章节编号
            section_num = re.search(r'^\d+\.?\d*', section_name)
            prefix = section_num.group().replace('.', '_') if section_num else ''
            filename = f"{prefix}_{chart_type}_{chart_id}.png"
        else:
            filename = f"{chart_type}_{chart_id}.png"
        
        # 图表存储路径
        filepath = os.path.join(self.output_dir, filename)
        
        # 根据图表类型调用相应的生成方法
        if chart_type == 'bar':
            self._generate_bar_chart(data, title, x_label, y_label, filepath)
        elif chart_type == 'line':
            self._generate_line_chart(data, title, x_label, y_label, filepath)
        elif chart_type == 'pie':
            self._generate_pie_chart(data, title, filepath)
        elif chart_type == 'scatter':
            self._generate_scatter_chart(data, title, x_label, y_label, filepath)
        elif chart_type == 'area':
            self._generate_area_chart(data, title, x_label, y_label, filepath)
        elif chart_type == 'heatmap':
            self._generate_heatmap_chart(data, title, x_label, y_label, filepath)
        elif chart_type == 'radar':
            self._generate_radar_chart(data, title, filepath)
        elif chart_type == 'bubble':
            self._generate_bubble_chart(data, title, x_label, y_label, filepath)
        else:
            logger.warning(f"不支持的图表类型: {chart_type}")
            return None
        
        # 返回相对路径，用于Markdown引用
        return os.path.join('charts', os.path.basename(filepath))
    
    def _generate_bar_chart(self, data, title, x_label, y_label, filepath):
        """生成柱状图"""
        try:
            # 创建图形和轴
            plt.figure(figsize=(10, 6))
            x_data = data.get('x', [])
            y_data = data.get('y', [])
            colors = data.get('colors', ['#2878B5'] * len(x_data))
            
            # 绘制柱状图
            plt.bar(x_data, y_data, color=colors)
            
            # 设置标题和标签
            plt.title(title, fontsize=14)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            
            # 设置网格和样式
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 旋转x轴标签，避免重叠
            plt.xticks(rotation=45, ha='right')
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"柱状图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成柱状图失败: {e}")
            return False
    
    def _generate_line_chart(self, data, title, x_label, y_label, filepath):
        """生成折线图"""
        try:
            plt.figure(figsize=(10, 6))
            
            # 处理多条线的情况
            if 'series' in data:
                for series in data['series']:
                    label = series.get('name', '')
                    x_values = series.get('x', [])
                    y_values = series.get('y', [])
                    color = series.get('color', '#2878B5')
                    marker = series.get('marker', 'o')
                    plt.plot(x_values, y_values, marker=marker, label=label, color=color)
                
                if any(series.get('name') for series in data['series']):
                    plt.legend()
            else:
                # 单条线的情况
                x_values = data.get('x', [])
                y_values = data.get('y', [])
                color = data.get('color', '#2878B5')
                marker = data.get('marker', 'o')
                plt.plot(x_values, y_values, marker=marker, color=color)
            
            # 设置标题和标签
            plt.title(title, fontsize=14)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            
            # 设置网格
            plt.grid(linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 处理x轴标签
            if len(data.get('x', [])) > 10 or any(len(str(x)) > 8 for x in data.get('x', [])):
                plt.xticks(rotation=45, ha='right')
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"折线图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成折线图失败: {e}")
            return False
    
    def _generate_pie_chart(self, data, title, filepath):
        """生成饼图"""
        try:
            plt.figure(figsize=(10, 8))
            
            # 提取数据
            labels = data.get('labels', [])
            values = data.get('values', [])
            colors = data.get('colors', None)
            explode = data.get('explode', None)
            
            # 绘制饼图
            wedges, texts, autotexts = plt.pie(
                values, 
                labels=None,  # 不在饼图中添加标签，而是使用图例
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                colors=colors,
                explode=explode,
                textprops={'fontsize': 12}
            )
            
            # 添加图例
            plt.legend(
                wedges, 
                labels,
                title="类别",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            # 设置标题
            plt.title(title, fontsize=14, pad=20)
            plt.axis('equal')  # 使饼图为正圆形
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"饼图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成饼图失败: {e}")
            return False
    
    def _generate_scatter_chart(self, data, title, x_label, y_label, filepath):
        """生成散点图"""
        try:
            plt.figure(figsize=(10, 6))
            
            # 提取数据
            x_values = data.get('x', [])
            y_values = data.get('y', [])
            colors = data.get('colors', '#2878B5')
            sizes = data.get('sizes', None)
            categories = data.get('categories', None)
            
            # 绘制散点图
            if categories:
                # 如果有分类，按类别绘制不同颜色的散点
                for i, cat in enumerate(set(categories)):
                    mask = [c == cat for c in categories]
                    x_cat = [x for x, m in zip(x_values, mask) if m]
                    y_cat = [y for y, m in zip(y_values, mask) if m]
                    plt.scatter(
                        x_cat, 
                        y_cat, 
                        label=cat,
                        alpha=0.7,
                        s=sizes
                    )
                plt.legend()
            else:
                # 没有分类，绘制单色散点
                plt.scatter(
                    x_values, 
                    y_values, 
                    c=colors,
                    alpha=0.7,
                    s=sizes
                )
            
            # 设置标题和标签
            plt.title(title, fontsize=14)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            
            # 设置网格
            plt.grid(linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"散点图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成散点图失败: {e}")
            return False
    
    def _generate_area_chart(self, data, title, x_label, y_label, filepath):
        """生成面积图"""
        try:
            plt.figure(figsize=(10, 6))
            
            # 处理多个区域的情况
            if 'series' in data:
                for series in data['series']:
                    label = series.get('name', '')
                    x_values = series.get('x', [])
                    y_values = series.get('y', [])
                    color = series.get('color', None)
                    alpha = series.get('alpha', 0.5)
                    plt.fill_between(x_values, y_values, alpha=alpha, label=label, color=color)
                
                if any(series.get('name') for series in data['series']):
                    plt.legend()
            else:
                # 单个区域的情况
                x_values = data.get('x', [])
                y_values = data.get('y', [])
                color = data.get('color', '#2878B5')
                alpha = data.get('alpha', 0.5)
                plt.fill_between(x_values, y_values, alpha=alpha, color=color)
            
            # 设置标题和标签
            plt.title(title, fontsize=14)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            
            # 设置网格
            plt.grid(linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 处理x轴标签
            if len(data.get('x', [])) > 10 or any(len(str(x)) > 8 for x in data.get('x', [])):
                plt.xticks(rotation=45, ha='right')
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"面积图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成面积图失败: {e}")
            return False
    
    def _generate_heatmap_chart(self, data, title, x_label, y_label, filepath):
        """生成热力图"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 提取数据
            matrix = data.get('matrix', [])
            x_labels = data.get('x_labels', [])
            y_labels = data.get('y_labels', [])
            cmap = data.get('cmap', 'viridis')
            
            # 创建热力图
            sns.heatmap(
                matrix, 
                annot=True, 
                fmt=".2f", 
                cmap=cmap,
                xticklabels=x_labels,
                yticklabels=y_labels,
                linewidths=0.5
            )
            
            # 设置标题和标签
            plt.title(title, fontsize=14, pad=20)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"热力图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成热力图失败: {e}")
            return False
    
    def _generate_radar_chart(self, data, title, filepath):
        """生成雷达图（多维度指标对比）
        
        Args:
            data (dict): 包含以下字段：
                - categories: 维度名称列表
                - series: 列表，每项包含name和values
                - max_value: 最大值（可选）
            title (str): 图表标题
            filepath (str): 保存路径
            
        Returns:
            bool: 是否成功生成
        """
        try:
            # 提取数据
            categories = data.get('categories', [])
            series = data.get('series', [])
            max_value = data.get('max_value', None)
            
            if not categories or not series:
                logger.warning("雷达图数据不足")
                return False
            
            # 计算角度
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # 闭合多边形
            
            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # 添加每个系列
            for serie in series:
                values = serie.get('values', [])
                if len(values) != N:
                    logger.warning(f"系列 {serie.get('name', '')} 的值数量与类别数量不匹配")
                    continue
                
                # 闭合多边形
                values_closed = values + [values[0]]
                
                # 绘制雷达图
                ax.plot(angles, values_closed, linewidth=2, label=serie.get('name', ''))
                ax.fill(angles, values_closed, alpha=0.25)
            
            # 设置刻度标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            
            # 设置y轴范围
            if max_value:
                ax.set_ylim(0, max_value)
            
            # 添加图例
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # 设置标题
            plt.title(title, fontsize=14, pad=20)
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"雷达图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成雷达图失败: {e}")
            return False
    
    def _generate_bubble_chart(self, data, title, x_label, y_label, filepath):
        """生成气泡图（三维数据可视化）
        
        Args:
            data (dict): 包含以下字段：
                - x: x轴数据
                - y: y轴数据
                - size: 气泡大小
                - labels: 气泡标签（可选）
                - colors: 气泡颜色（可选）
            title (str): 图表标题
            x_label (str): x轴标签
            y_label (str): y轴标签
            filepath (str): 保存路径
            
        Returns:
            bool: 是否成功生成
        """
        try:
            # 提取数据
            x_values = data.get('x', [])
            y_values = data.get('y', [])
            sizes = data.get('size', [100] * len(x_values))  # 默认大小
            labels = data.get('labels', None)
            colors = data.get('colors', None)
            
            if not x_values or not y_values:
                logger.warning("气泡图数据不足")
                return False
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 绘制气泡图
            scatter = plt.scatter(
                x_values, 
                y_values, 
                s=sizes, 
                c=colors,
                alpha=0.6, 
                edgecolors='w', 
                linewidth=0.5
            )
            
            # 添加气泡标签
            if labels:
                for i, label in enumerate(labels):
                    plt.annotate(
                        label,
                        (x_values[i], y_values[i]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        alpha=0.8
                    )
            
            # 设置标题和标签
            plt.title(title, fontsize=14)
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
            
            # 添加图例或颜色条
            if colors is not None and len(set(colors)) > 1:
                plt.colorbar(scatter, label='Value')
            
            # 设置网格
            plt.grid(linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"气泡图已保存: {filepath}")
            return True
        except Exception as e:
            logger.error(f"生成气泡图失败: {e}")
            return False
    
    def generate_chart_description(self, chart_config, chart_path):
        """根据图表配置和模板生成图表描述
        
        Args:
            chart_config (dict): 图表配置，包含描述模板和数据
            chart_path (str): 图表文件路径
            
        Returns:
            str: 图表描述的Markdown文本
        """
        try:
            # 准备基本描述
            chart_type = chart_config['type']
            title = chart_config.get('title', '图表')
            template = chart_config.get('description_template', '该图表展示了相关数据分析结果。')
            
            # 准备模板变量
            template_vars = {}
            data = chart_config['data']
            
            # 根据图表类型填充模板变量
            if chart_type == 'line':
                # 处理折线图
                if 'series' in data:
                    # 多条线的情况，使用最后一条线的数据
                    last_series = data['series'][-1]
                    x_values = last_series.get('x', [])
                    y_values = last_series.get('y', [])
                else:
                    # 单条线
                    x_values = data.get('x', [])
                    y_values = data.get('y', [])
                
                # 计算趋势
                if len(y_values) >= 2:
                    start_val = y_values[0]
                    end_val = y_values[-1]
                    if end_val > start_val * 1.2:
                        trend = "明显上升"
                    elif end_val > start_val:
                        trend = "小幅上升"
                    elif end_val < start_val * 0.8:
                        trend = "明显下降"
                    elif end_val < start_val:
                        trend = "小幅下降"
                    else:
                        trend = "相对稳定"
                else:
                    trend = "波动"
                
                # 填充变量
                if x_values:
                    template_vars['year_start'] = x_values[0]
                    template_vars['year_end'] = x_values[-1]
                template_vars['trend_description'] = trend
                
                # 处理预测相关变量
                if 'series' in data and len(data['series']) >= 3:
                    opt_series = data['series'][1]  # 乐观预测
                    con_series = data['series'][2]  # 保守预测
                    
                    template_vars['start_year'] = opt_series['x'][0]
                    template_vars['end_year'] = opt_series['x'][-1]
                    template_vars['optimistic_value'] = f"{opt_series['y'][-1]:.1f}"
                    template_vars['conservative_value'] = f"{con_series['y'][-1]:.1f}"
                    
                    # 计算年均复合增长率(CAGR)
                    years = len(opt_series['x']) - 1
                    if years > 0:
                        opt_cagr = ((opt_series['y'][-1] / opt_series['y'][0]) ** (1/years) - 1) * 100
                        con_cagr = ((con_series['y'][-1] / con_series['y'][0]) ** (1/years) - 1) * 100
                        template_vars['min_cagr'] = f"{min(opt_cagr, con_cagr):.1f}"
                        template_vars['max_cagr'] = f"{max(opt_cagr, con_cagr):.1f}"
            
            elif chart_type == 'bar':
                # 处理柱状图
                x_values = data.get('x', [])
                y_values = data.get('y', [])
                
                if x_values and y_values:
                    # 查找最大值和最小值
                    max_index = y_values.index(max(y_values))
                    min_index = y_values.index(min(y_values))
                    
                    template_vars['year_start'] = x_values[0]
                    template_vars['year_end'] = x_values[-1]
                    template_vars['max_year'] = x_values[max_index]
                    template_vars['min_year'] = x_values[min_index]
                    template_vars['max_rate'] = f"{y_values[max_index]:.1f}"
                    template_vars['min_rate'] = f"{y_values[min_index]:.1f}"
                    
                    # 区域图表特殊变量
                    if '区域' in title or '地区' in title:
                        template_vars['top_region'] = x_values[max_index]
                        template_vars['top_share'] = f"{y_values[max_index]:.1f}"
                        # 假设增长最快的区域
                        growth_index = (max_index + 1) % len(x_values)
                        template_vars['growth_region'] = x_values[growth_index]
                    
                    # 计算趋势
                    if len(y_values) >= 3:
                        if all(y_values[i] <= y_values[i+1] for i in range(len(y_values)-1)):
                            trend = "持续上升"
                        elif all(y_values[i] >= y_values[i+1] for i in range(len(y_values)-1)):
                            trend = "持续下降"
                        elif y_values[-1] > y_values[0]:
                            trend = "总体上升但有波动"
                        elif y_values[-1] < y_values[0]:
                            trend = "总体下降但有波动"
                        else:
                            trend = "波动但总体稳定"
                        
                        template_vars['trend_description'] = trend
            
            elif chart_type == 'pie':
                # 处理饼图
                labels = data.get('labels', [])
                values = data.get('values', [])
                
                if labels and values:
                    max_index = values.index(max(values))
                    template_vars['top_company'] = labels[max_index]
                    template_vars['top_share'] = f"{values[max_index]:.1f}"
                    template_vars['top_count'] = min(5, len(labels))
                    
                    # 计算前几名的总份额
                    top_n = min(5, len(values))
                    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
                    top_indices = sorted_indices[:top_n]
                    total_top = sum(values[i] for i in top_indices)
                    template_vars['total_share'] = f"{total_top:.1f}"
            
            elif chart_type == 'heatmap':
                # 处理热力图
                matrix = data.get('matrix', [])
                x_labels = data.get('x_labels', [])
                y_labels = data.get('y_labels', [])
                
                if len(matrix) > 0 and len(x_labels) > 0 and len(y_labels) > 0:
                    # 找出最高分和最低分的位置
                    max_val = float('-inf')
                    min_val = float('inf')
                    max_i, max_j = 0, 0
                    min_i, min_j = 0, 0
                    
                    for i in range(len(matrix)):
                        for j in range(len(matrix[i])):
                            if matrix[i][j] > max_val:
                                max_val = matrix[i][j]
                                max_i, max_j = i, j
                            if matrix[i][j] < min_val:
                                min_val = matrix[i][j]
                                min_i, min_j = i, j
                    
                    template_vars['top_company'] = y_labels[max_i]
                    template_vars['top_category'] = x_labels[max_j]
                    template_vars['weak_category'] = x_labels[min_j]
            
            # 使用模板生成描述
            # 替换模板中的变量
            description = template
            for key, value in template_vars.items():
                placeholder = f"{{{key}}}"
                if placeholder in description:
                    description = description.replace(placeholder, str(value))
            
            # 生成完整的图表描述（包含图片引用）
            markdown_text = f"\n\n![{title}]({chart_path})\n\n**图表说明：** {description}\n\n"
            return markdown_text
            
        except Exception as e:
            logger.error(f"生成图表描述失败: {e}")
            # 返回基本描述
            return f"\n\n![{chart_config.get('title', '图表')}]({chart_path})\n\n"
    
    def auto_chart_mapping(self, data_metrics, section_name=None):
        """自动映射数据到合适的图表类型
        
        Args:
            data_metrics (dict): 数据指标字典
            section_name (str, optional): 章节名称
            
        Returns:
            list: 图表配置列表
        """
        chart_configs = []
        
        try:
            # 如果没有数据，返回空列表
            if not data_metrics or not any(data_metrics.values()):
                logger.warning(f"章节 {section_name} 没有可用数据")
                return []
            
            # 1. 处理市场规模数据 - 使用柱状图或折线图
            if 'market_size' in data_metrics and len(data_metrics['market_size']) >= 3:
                # 提取市场规模数据
                market_size_data = data_metrics['market_size']
                
                # 创建柱状图配置
                bar_data = {
                    'x': [f"数据{i+1}" for i in range(len(market_size_data))],
                    'y': [float(re.findall(r'\d+\.?\d*', item['value'])[0]) if re.findall(r'\d+\.?\d*', item['value']) else 0 for item in market_size_data[:5]],
                    'colors': ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de'][:len(market_size_data[:5])]
                }
                
                bar_config = {
                    'type': 'bar',
                    'data': bar_data,
                    'title': '市场规模数据对比',
                    'x_label': '数据来源',
                    'y_label': '市场规模',
                    'description_template': '该图展示了来自不同来源的{name}市场规模数据对比。'
                }
                
                chart_configs.append(bar_config)
            
            # 2. 处理增长率数据 - 使用折线图
            if 'growth_rate' in data_metrics and len(data_metrics['growth_rate']) >= 2:
                # 提取增长率数据
                growth_data = data_metrics['growth_rate']
                
                # 创建模拟时间序列
                years = [(datetime.now().year - 2 + i) for i in range(5)]
                
                # 创建两个增长曲线（乐观和保守）
                optimistic_values = [100]
                conservative_values = [100]
                
                # 提取增长率数值
                growth_rates = []
                for item in growth_data:
                    match = re.search(r'(\d+\.?\d*)%', item['value'])
                    if match:
                        rate = float(match.group(1)) / 100
                        growth_rates.append(rate)
                
                # 使用提取的增长率或默认值
                if growth_rates:
                    avg_rate = sum(growth_rates) / len(growth_rates)
                    high_rate = max(growth_rates)
                    low_rate = min(growth_rates)
                else:
                    avg_rate = 0.10
                    high_rate = 0.15
                    low_rate = 0.05
                
                # 生成未来值
                for _ in range(len(years) - 1):
                    optimistic_values.append(optimistic_values[-1] * (1 + high_rate))
                    conservative_values.append(conservative_values[-1] * (1 + low_rate))
                
                # 创建折线图数据
                line_data = {
                    'series': [
                        {
                            'name': '乐观预测',
                            'x': [str(year) for year in years],
                            'y': optimistic_values,
                            'color': '#91cc75'
                        },
                        {
                            'name': '保守预测',
                            'x': [str(year) for year in years],
                            'y': conservative_values,
                            'color': '#fac858'
                        }
                    ]
                }
                
                line_config = {
                    'type': 'line',
                    'data': line_data,
                    'title': '市场增长趋势预测',
                    'x_label': '年份',
                    'y_label': '市场规模指数',
                    'description_template': '该图展示了基于当前增长率的市场规模预测趋势，乐观预测采用{high_rate:.1%}的年增长率，保守预测采用{low_rate:.1%}的年增长率。'
                }
                
                chart_configs.append(line_config)
            
            # 3. 处理市场份额数据 - 使用饼图
            if 'market_share' in data_metrics and len(data_metrics['market_share']) >= 3:
                # 提取市场份额数据
                share_data = data_metrics['market_share']
                
                # 提取公司名称和份额
                companies = []
                shares = []
                
                for item in share_data:
                    # 尝试提取公司名和份额
                    company_match = re.search(r'([A-Za-z\u4e00-\u9fa5][A-Za-z\u4e00-\u9fa5\s]{0,20}(?:公司|企业|集团|品牌|Corp|Inc|Company|Technologies|Tech))', item['context'])
                    share_match = re.search(r'(\d+\.?\d*)%', item['value'])
                    
                    if company_match and share_match:
                        companies.append(company_match.group(1))
                        shares.append(float(share_match.group(1)))
                
                # 如果提取到足够的数据
                if len(companies) >= 3 and len(shares) >= 3:
                    # 数据排序
                    sorted_data = sorted(zip(companies, shares), key=lambda x: x[1], reverse=True)
                    companies, shares = zip(*sorted_data)
                    
                    # 如果公司过多，合并小份额
                    if len(companies) > 6:
                        others_share = sum(shares[5:])
                        companies = companies[:5] + ('其他',)
                        shares = shares[:5] + (others_share,)
                    
                    # 创建饼图数据
                    pie_data = {
                        'labels': companies,
                        'values': shares,
                        'colors': ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4'][:len(companies)]
                    }
                    
                    pie_config = {
                        'type': 'pie',
                        'data': pie_data,
                        'title': '市场份额分布',
                        'description_template': '该图展示了主要企业在市场中的份额分布，其中{top_company}占据最大份额为{top_share:.1f}%。'
                    }
                    
                    chart_configs.append(pie_config)
            
            # 4. 处理波特五力分析 - 使用雷达图
            if section_name and "波特五力" in section_name:
                # 为波特五力分析创建雷达图
                radar_data = {
                    'categories': ['现有竞争者威胁', '供应商议价能力', '购买者议价能力', '替代品威胁', '新进入者威胁'],
                    'series': [
                        {
                            'name': '竞争强度',
                            'values': [0.7, 0.4, 0.6, 0.5, 0.3]  # 示例值，理想情况下应从数据中提取
                        }
                    ],
                    'max_value': 1.0
                }
                
                # 尝试从数据中提取实际的五力评分
                if any('五力' in item.get('context', '') for items in data_metrics.values() for item in items):
                    # 这里可以添加更复杂的逻辑来提取实际的五力评分数据
                    pass
                
                radar_config = {
                    'type': 'radar',
                    'data': radar_data,
                    'title': '波特五力分析',
                    'description_template': '该雷达图展示了行业五种竞争力量的相对强度，从图中可以看出，{strongest_force}是最强的竞争力量，而{weakest_force}相对较弱。'
                }
                
                chart_configs.append(radar_config)
            
            # 5. 处理竞争格局分析 - 使用气泡图
            if section_name and "竞争格局" in section_name:
                # 为竞争格局分析创建气泡图，展示公司规模、增长率和市场份额
                bubble_data = {
                    'x': [0.05, 0.12, 0.08, 0.15, 0.07, 0.1],  # 增长率
                    'y': [200, 150, 300, 100, 250, 180],      # 营收规模
                    'size': [1500, 1000, 2500, 800, 2000, 1200],  # 市场份额（气泡大小）
                    'labels': ['公司A', '公司B', '公司C', '公司D', '公司E', '公司F'],
                    'colors': ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272']
                }
                
                # 尝试从数据中提取实际的公司数据
                if 'market_share' in data_metrics and len(data_metrics['market_share']) >= 3:
                    # 这里可以添加逻辑来提取真实公司数据
                    pass
                
                bubble_config = {
                    'type': 'bubble',
                    'data': bubble_data,
                    'title': '主要企业营收规模与增长率对比',
                    'x_label': '增长率',
                    'y_label': '营收规模（亿元）',
                    'description_template': '该气泡图展示了主要企业的营收规模、增长率和市场份额（气泡大小），可以直观地比较不同企业的竞争地位。'
                }
                
                chart_configs.append(bubble_config)
            
            # 6. 特定章节的热力图 - 用于矩阵分析
            if section_name and any(keyword in section_name for keyword in ["产品组合", "BCG矩阵"]):
                # 创建BCG矩阵热力图
                heatmap_data = {
                    'matrix': [
                        [0.8, 0.6, 0.3, 0.1],
                        [0.7, 0.9, 0.5, 0.2],
                        [0.4, 0.7, 0.8, 0.3],
                        [0.2, 0.3, 0.6, 0.7]
                    ],
                    'x_labels': ['明星业务', '现金牛业务', '问题业务', '瘦狗业务'],
                    'y_labels': ['市场增长率', '市场份额', '盈利能力', '投资需求']
                }
                
                heatmap_config = {
                    'type': 'heatmap',
                    'data': heatmap_data,
                    'title': '产品组合矩阵分析',
                    'x_label': '业务类型',
                    'y_label': '评估维度',
                    'description_template': '该热力图展示了不同类型业务在各个评估维度上的表现，颜色越深表示表现越好。'
                }
                
                chart_configs.append(heatmap_config)
            
            logger.info(f"为章节 {section_name} 生成了 {len(chart_configs)} 个图表配置")
            return chart_configs
            
        except Exception as e:
            logger.error(f"自动映射图表失败: {e}")
            return []

# 测试代码
if __name__ == "__main__":
    chart_gen = ChartGenerator()
    
    # 测试生成柱状图
    bar_data = {
        'x': ['2020', '2021', '2022', '2023', '2024'],
        'y': [12.5, 15.8, 18.2, 22.6, 25.8]
    }
    
    bar_chart_path = chart_gen.generate_chart(
        data=bar_data,
        chart_type='bar',
        title='市场规模年度变化',
        x_label='年份',
        y_label='市场规模(亿元)',
        section_name='3. 市场规模'
    )
    
    print(f"生成的柱状图保存在: {bar_chart_path}")
    
    # 测试生成饼图
    pie_data = {
        'labels': ['企业A', '企业B', '企业C', '企业D', '其他'],
        'values': [35, 25, 15, 10, 15]
    }
    
    pie_chart_path = chart_gen.generate_chart(
        data=pie_data,
        chart_type='pie',
        title='市场份额分布',
        section_name='6. 竞争格局分析'
    )
    
    print(f"生成的饼图保存在: {pie_chart_path}") 