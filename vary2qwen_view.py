import json
import cv2
import os
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # SimHei 是一种常用的中文字体
rcParams['axes.unicode_minus'] = False    # 防止负号显示为方块

def visualize_grounding(data):
    """
    可视化grounding任务的标注文件，显示图像上的矩形框和类别标签。
    """
    image_path = data["images"][0]  # 获取图像路径
    image = cv2.imread(image_path)  # 读取图像

    # 创建Matplotlib的窗口
    fig, ax = plt.subplots(figsize=(10, 7))

    # 读取图像并转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image_rgb)
    ax.axis('off')  # 不显示坐标轴

    # 绘制矩形框并显示类别标签
    objects = data["objects"]
    for obj in objects:
        caption = obj['caption']  # 获取类别
        for bbox in obj['bbox']:
            x1, y1, x2, y2 = bbox  # 获取x1, y1, x2, y2
            width = x2 - x1  # 宽度
            height = y2 - y1  # 高度

            # 绘制矩形框
            ax.add_patch(plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none'))

            # 在矩形框上方显示类别标签
            ax.text(x1, y1 - 10, caption, fontsize=10, color='red', ha="left", va="bottom", weight='bold')

    plt.tight_layout()
    plt.show()

def restore_bbox(img_data, bbox):
    img_height, img_width, _ = img_data.shape
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = x1/1000*img_width, y1/1000*img_height, x2/1000*img_width, y2/1000*img_height
    return [int(x1), int(y1), int(x2), int(y2)]

def visualize_region(data):
    """
    可视化数据中的图像和对应的多个区域描述。
    
    data: dict，包含query, response, image字段。
    """
    # 获取图像路径
    image_path = data["image"]
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 提取所有的区域框 (bbox)
    query = data["query"]
    box_matches = re.findall(r'<box>\[\[(\d+), (\d+), (\d+), (\d+)\]\]</box>', query)
    
    # 绘制所有矩形框
    for box_match in box_matches:
        x1, y1, x2, y2 = map(int, box_match)  # 提取坐标
        x1, y1, x2, y2 = restore_bbox(image, [x1, y1, x2, y2])
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 获取描述信息
    description = data["response"]
    
    # 创建一个图像窗口
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 在左边显示图像
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV默认读取为BGR，需要转换为RGB
    ax[0].axis('off')  # 不显示坐标轴
    
    # 在右边显示描述文本
    ax[1].axis('off')  # 不显示坐标轴
    
    # 为文本添加换行
    wrapped_text = "\n".join(description[i:i+20] for i in range(0, len(description), 20))  # 每行最多显示40个字符
    
    ax[1].text(0.1, 0.9, wrapped_text, fontsize=12, wrap=True, ha="left", va="top", transform=ax[1].transAxes)
    
    # 显示图像
    plt.tight_layout()
    plt.show()

def load_and_visualize_jsonl(jsonl_path):
    """
    加载标注文件并根据任务类型进行可视化
    """
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # 逐行加载每个JSON对象
            if "objects" in data:  # 这是grounding任务
                visualize_grounding(data)
            elif "image" in data:  # 这是region任务
                visualize_region(data)
                pass
            else:
                print(f"无法识别的任务类型，跳过文件 {jsonl_path}")

def visualize_all_jsonl_in_folder(folder_path):
    """
    遍历文件夹中的所有标注文件并可视化
    """
    # 获取文件夹中的所有JSONL文件
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    
    for jsonl_file in jsonl_files:
        jsonl_path = os.path.join(folder_path, jsonl_file)
        print(f"正在可视化文件: {jsonl_path}")
        load_and_visualize_jsonl(jsonl_path)

# 示例：可视化文件夹中的所有标注文件
folder_path = 'D:/code/data_extract/save/new/'  # 请替换为你的文件夹路径
visualize_all_jsonl_in_folder(folder_path)
