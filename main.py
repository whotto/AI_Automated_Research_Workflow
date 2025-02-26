#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
市场研究工作流主脚本
用于执行完整的市场研究工作流程
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from research_workflow import ResearchWorkflow

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('research_workflow.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主函数，执行研究工作流"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='市场研究工作流')
    parser.add_argument('query', type=str, help='研究主题，例如"显微镜市场"')
    parser.add_argument('--skip-spider', action='store_true', help='跳过爬虫步骤，直接使用已有数据')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("启用详细日志模式")
    
    # 加载环境变量
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("未找到 OPENAI_API_KEY 环境变量，请在 .env 文件中设置")
        return 1
    
    try:
        logger.info("=" * 50)
        logger.info(f"开始执行市场研究工作流，研究主题: {args.query}")
        logger.info("=" * 50)
        
        # 创建工作流对象
        workflow = ResearchWorkflow()
        
        # 运行工作流
        result = workflow.run_pipeline(args.query)
        
        # 处理结果
        if result.get('status') == 'success':
            logger.info("=" * 50)
            logger.info(f"研究工作流成功完成！")
            logger.info(f"报告文件: {result.get('report_file')}")
            logger.info(f"处理的数据项数: {result.get('data_count')}")
            logger.info("=" * 50)
            
            # 尝试打开报告文件
            report_file = result.get('report_file')
            if report_file and os.path.exists(report_file):
                try:
                    if sys.platform == 'darwin':  # macOS
                        os.system(f'open "{report_file}"')
                    elif sys.platform == 'win32':  # Windows
                        os.system(f'start "" "{report_file}"')
                    else:  # Linux
                        os.system(f'xdg-open "{report_file}"')
                    logger.info(f"已尝试打开报告文件")
                except Exception as e:
                    logger.warning(f"尝试打开报告文件失败: {e}")
            
            return 0
        else:
            logger.error("=" * 50)
            logger.error(f"研究工作流失败: {result.get('message')}")
            logger.error("=" * 50)
            return 1
            
    except KeyboardInterrupt:
        logger.warning("用户中断执行")
        return 1
    except Exception as e:
        logger.exception(f"执行过程中发生错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 