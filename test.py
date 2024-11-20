import json
import os
import random
import re

# 判断是否是grounding任务
def is_grounding_task(conversations):
    for convo in conversations:
        if convo["from"] == "gpt":
            if re.search(r"<ref>.*</ref>", convo["value"]) and re.search(r"<box>.*</box>", convo["value"]):
                return True
    return False

# 转换为grounding任务格式
def convert_vary_2grounding(original_data):
    target_data = {
        "query": "找到 <ref-object>",
        "response": "<bbox>",
        "images": [],
        "objects": []
    }
    target_data["images"].append(original_data["image"])
    
    for conversation in original_data["conversations"]:
        if conversation["from"] == "gpt":
            object_info = {
                "caption": conversation["value"].split('<ref>')[1].split('</ref>')[0],
                "bbox": [],
                "bbox_type": "real",
                "image": 0
            }
            bbox_info = conversation["value"].split('<box>')[1].split('</box>')[0]
            object_info["bbox"] = [list(map(int, bbox_info.strip('[]').split(',')))]
            target_data["objects"].append(object_info)
    
    return target_data

# 转换为VQA任务格式
def convert_vary_2cap(data):
    image_path = data["image"]
    human_text = next(convo["value"].replace("<image>/n", "").strip() for convo in data["conversations"] if convo["from"] == "human")
    gpt_text = next(convo["value"] for convo in data["conversations"] if convo["from"] == "gpt")
    
    new_item = {
        "query": f"<image>{human_text}",
        "response": gpt_text,
        "images": [image_path]
    }
    return new_item

# 提取问答对
def extract_qa_pairs(data):
    qa_pairs = []
    for item in data:
        image_path = item["image"]
        conversations = item["conversations"]
        
        for i in range(0, len(conversations), 2):
            if i + 1 < len(conversations):
                question = conversations[i]["value"].replace("<image>\n", "").strip()
                answer = conversations[i + 1]["value"].strip()
                
                qa_pairs.append({
                    "query": question,
                    "response": answer,
                    "image": image_path
                })
    
    return qa_pairs

# 保存数据为JSONL文件
def save_data_as_jsonl(data, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

# 主转换函数
def process_dataset(img_dir_labels, output_dir, split_ratio=0.8):
    for dataset in img_dir_labels:
        img_dir = dataset['img_dir']
        label_dir = dataset['labels']
        
        train_data = []
        val_data = []
        
        # 获取所有标注文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        
        random.shuffle(label_files)  # 打乱文件顺序
        
        train_size = int(len(label_files) * split_ratio)  # 计算训练集大小
        train_files = label_files[:train_size]
        val_files = label_files[train_size:]
        
        # 处理训练集文件
        for label_file in train_files:
            label_path = os.path.join(label_dir, label_file)
            try:
                with open(label_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    if is_grounding_task(data):
                        converted_data = convert_vary_2grounding(data)
                    else:
                        converted_data = extract_qa_pairs(data)
                    
                    train_data.append(converted_data)
                    
            except Exception as e:
                print(f"处理训练集文件 {label_path} 时发生错误: {e}")
        
        # 处理验证集文件
        for label_file in val_files:
            label_path = os.path.join(label_dir, label_file)
            try:
                with open(label_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    if is_grounding_task(data):
                        converted_data = convert_vary_2grounding(data)
                    else:
                        converted_data = extract_qa_pairs(data)
                    
                    val_data.append(converted_data)
                    
            except Exception as e:
                print(f"处理验证集文件 {label_path} 时发生错误: {e}")
        
        # 保存训练集数据
        train_output_path = os.path.join(output_dir, 'train.jsonl')
        if train_data:
            save_data_as_jsonl(train_data, train_output_path)
            print(f"已保存训练集数据到 {train_output_path}")
        
        # 保存验证集数据
        val_output_path = os.path.join(output_dir, 'val.jsonl')
        if val_data:
            save_data_as_jsonl(val_data, val_output_path)
            print(f"已保存验证集数据到 {val_output_path}")

# 测试数据列表
img_dir_labels = [
    {"img_dir": "path/to/images/dataset1", "labels": "path/to/labels/dataset1"},
    {"img_dir": "path/to/images/dataset2", "labels": "path/to/labels/dataset2"}
]

output_dir = 'path/to/output'  # 设定输出路径
split_ratio = 0.8  # 80% 训练集，20% 验证集

# 执行处理
process_dataset(img_dir_labels, output_dir, split_ratio)
