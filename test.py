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
def convert_vary_2grounding(original_data, images_dir):
    target_data = {
        "query": "找到 <ref-object>",
        "response": "<bbox>",
        "images": [],
        "objects": []
    }
    
    # 更新图片路径为绝对路径
    image_path = os.path.join(images_dir, original_data["image"])
    target_data["images"].append(image_path)
    
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
def convert_vary_2cap(data, images_dir):
    image_path = os.path.join(images_dir, data["image"])
    human_text = next(convo["value"].replace("<image>/n", "").strip() for convo in data["conversations"] if convo["from"] == "human")
    gpt_text = next(convo["value"] for convo in data["conversations"] if convo["from"] == "gpt")
    
    new_item = {
        "query": f"<image>{human_text}",
        "response": gpt_text,
        "images": [image_path]
    }
    return new_item

# 提取问答对
def extract_qa_pairs(data, images_dir):
    qa_pairs = []
    for item in data:
        image_path = os.path.join(images_dir, item["image"])  # 将相对路径转换为绝对路径
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
    for dataset_name, dataset in img_dir_labels.items():
        img_dir = dataset['images']
        label_dir = dataset['annotations']
        
        train_data = []
        val_data = []
        all_data = []
        # 获取所有标注文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        
        random.shuffle(label_files)  # 打乱文件顺序
        
        
        # train_files = label_files[:train_size]
        # val_files = label_files[train_size:]
        
        # 处理训练集文件
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            try:
                with open(label_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    if is_grounding_task(data):
                        if(valid_grounding(label_path)):
                            converted_data = convert_vary_2grounding(data, img_dir)
                    else:
                        converted_data = extract_qa_pairs(data, img_dir)
                    
                    all_data.append(converted_data)       
            except Exception as e:
                print(f"处理训练集文件 {label_path} 时发生错误: {e}")
        
        train_size = int(len(all_data) * split_ratio)  # 计算训练集大小
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]
        
        # 保存训练集数据
        train_output_path = os.path.join(output_dir, f'{dataset_name}_train.jsonl')
        if train_data:
            save_data_as_jsonl(train_data, train_output_path)
            print(f"已保存训练集数据到 {train_output_path}")
        
        # 保存验证集数据
        val_output_path = os.path.join(output_dir, f'{dataset_name}_val.jsonl')
        if val_data:
            save_data_as_jsonl(val_data, val_output_path)
            print(f"已保存验证集数据到 {val_output_path}")


def valid_grounding(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        
        question = json_data['conversations'][0]['value']
        answer = json_data['conversations'][1]['value']

        # 检查错误的回答：包含 'None' 或 '不存在此类别'
        if 'None' in answer or '不存在此类别' in answer:
            print(f"{json_path} 回答错误")
            return False
        
        # 正则表达式(回答标签)
        pattern = r'<ref>(.*?)<\/ref>'
        matches = re.findall(pattern, answer)
        for i in matches:
            if i not in question:
                print(f"{json_path} 输入目标与输出目标不匹配")
                return False
        
        return True
    
    except Exception as e:
        print(f"处理文件 {json_path} 时发生错误: {e}")
        return False



# 测试数据列表
img_dir_labels = {
    'alg_base_vqa':{
        'images':"/data2/liangqh/Datasets/alg_base/",
        'annotations':"/data2/liangqh/Datasets/alg_base/alg_base_vqa/",
    },
    'tower_data':{
        'images': "/data2/liangqh/Datasets/Tower_dataset/",
        'annotations': "/data2/liangqh/Datasets/Tower_dataset/labels/",
    },
    'alg_base_Cap':{
        'images':"/data2/liangqh/Datasets/alg_base/",
        'annotations':"/data2/liangqh/Datasets/alg_base/GLM4v_captions/",
    },
    "Tower_bigdata":{
        "images":"/data2/liangqh/Datasets/Tower_dataset/",
        "annotations":"/data2/liangqh/Datasets/Tower_dataset/bigdata_json/label_json_box/",
    },
    "Tower_bigdata_cap":{
        "images":"/data2/liangqh/Datasets/Tower_dataset/bigdata/",
        "annotations":"/data2/liangqh/Datasets/Tower_dataset/bigdata_json/bigdata_cap/",
    },
    "Tower_data_1":{
        "images":"/data2/liangqh/Datasets/Tower_dataset/Tower_data_1/images/",
        "annotations":"/data2/liangqh/Datasets/Tower_dataset/Tower_data_1/jsons/"
    },
    "Tower_data_1_ref":{
        "images":"/data3/liangqh/Datasets/Tower_data/Tower_data_2/images/",
        "annotations":"/data3/liangqh/Datasets/Tower_data/Tower_data_2/jsons/",
    },
    "alg_base_regionCap":{
        "images":"/data2/liangqh/Datasets/alg_base/",
        "annotations":"/data2/liangqh/Datasets/alg_base/algbase_regionCap/",
    },
    "Tower_bigdata_regionCap":{
        "images":"/data2/liangqh/Datasets/Tower_dataset/bigdata/",
        "annotations":"/data2/liangqh/Datasets/Tower_dataset/bigdata_json/bigdata_regionCap/",
    },
}

output_dir = '/path/to/output'  # 设定输出路径
split_ratio = 0.8  # 80% 训练集，20% 验证集

# 执行处理
process_dataset(img_dir_labels, output_dir, split_ratio)
