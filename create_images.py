from PIL import Image, ImageDraw, ImageFont
import os
from math import sin, cos, pi

def create_gradient_background(width, height, color1, color2):
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    
    for y in range(height):
        r = r1 + (r2 - r1) * y // height
        g = g1 + (g2 - g1) * y // height
        b = b1 + (b2 - b1) * y // height
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    return image

def draw_hexagon(draw, center, size, color, width=2):
    points = []
    for i in range(6):
        angle = i * pi / 3
        points.append((center[0] + size * cos(angle),
                      center[1] + size * sin(angle)))
    draw.line(points + [points[0]], fill=color, width=width)

def create_banner(output_path, width=1200, height=400):
    # 创建渐变背景
    image = create_gradient_background(width, height, '#1a237e', '#0277bd')
    draw = ImageDraw.Draw(image)
    
    # 添加装饰性六边形
    for i in range(8):
        size = 30 + i * 15
        draw_hexagon(draw, (100, height//2), size, '#ffffff', width=1)
        draw_hexagon(draw, (width-100, height//2), size, '#ffffff', width=1)
    
    # 添加主标题
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 120)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 36)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
    
    title = "智研数析"
    subtitle = "AI驱动的市场研究解决方案"
    
    # 绘制主标题
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    x = (width - title_width) // 2
    y = height // 3
    draw.text((x, y), title, fill='white', font=title_font)
    
    # 绘制副标题
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    x = (width - subtitle_width) // 2
    y = y + 100
    draw.text((x, y), subtitle, fill='#e3f2fd', font=subtitle_font)
    
    image.save(output_path)

def create_workflow(output_path, width=1200, height=250):
    # 创建背景
    image = Image.new('RGB', (width, height), '#ffffff')
    draw = ImageDraw.Draw(image)
    
    # 添加渐变背景
    for y in range(height):
        alpha = int(255 * (1 - y/height))
        draw.line([(0, y), (width, y)], fill=(240, 244, 255))
    
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 36)
        step_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
    except:
        title_font = ImageFont.load_default()
        step_font = ImageFont.load_default()
    
    steps = ["需求分析", "数据采集", "数据处理", "内容生成", "可视化创建", "整合报告"]
    
    # 绘制步骤
    step_width = width // (len(steps) + 1)
    y = height // 2
    
    for i, step in enumerate(steps):
        x = step_width * (i + 1)
        
        # 绘制圆圈
        circle_radius = 40
        draw.ellipse([x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius],
                     outline='#1a237e', width=2)
        
        # 绘制文字
        text_bbox = draw.textbbox((0, 0), step, font=step_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x - text_width//2
        draw.text((text_x, y-12), step, fill='#1a237e', font=step_font)
        
        # 绘制连接线
        if i < len(steps) - 1:
            draw.line([x+circle_radius+10, y, x+step_width-circle_radius-10, y],
                     fill='#1a237e', width=2)
    
    image.save(output_path)

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("assets/images", exist_ok=True)
    
    # 创建图片
    create_banner("assets/images/banner.png")
    create_workflow("assets/images/workflow.png")
    print("图片已生成在 assets/images 目录下")
