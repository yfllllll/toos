import json
import os
import random
import re
import cv2
from collections import defaultdict

def od_restore_bbox(bboxes, image_h_w, BOX_SCALE = 999):
    restored_bboxes = {}
    height, width = image_h_w
    for classname, boxes in bboxes.items():
        restored_boxes = []
        for bbox in boxes:
            # 按比例还原到原始大小
            bbox = [
                int(bbox[0]/BOX_SCALE * max(height, width)),
                int(bbox[1]/BOX_SCALE * max(height, width)),
                int(bbox[2]/BOX_SCALE * max(height, width)),
                int(bbox[3]/BOX_SCALE * max(height, width)),
            ]
            if height == width:
                pass
            elif height < width:
                delta=(width-height)//2
                bbox[1]-= delta
                bbox[3]-= delta
            else:
                delta=(height - width)// 2
                bbox[0]-= delta
                bbox[2]-= delta
            for i in range(len(bbox)):
                if bbox[i]< 0:
                    bbox[i]= 0
            restored_boxes.append(bbox)
        restored_bboxes[classname] = restored_boxes
    return restored_bboxes

def dumy_obj(objects):
    output_str = ""
    for k, v in objects.items():
        output_str = output_str + "<ref>"+ k + "</ref>" + "<box>" + json.dumps(v) + '</box>,'
    output_str = output_str[:-1]
    return output_str
def extract_obj(
    grounded_caption: str,
    grounded_pattern: str = r'<.*?>.*?<.*?>',):
    
   
    REF_START_TAG = '<ref>'
    REF_END_TAG = '</ref>'
    BOX_START_TAG = '<box>'
    BOX_END_TAG = '</box>'
    REL_START_TAG = '<pred>'
    REL_END_TAG = '</pred>'
    
    objects = defaultdict(list)
    relations = defaultdict(list)
    clean_caption = grounded_caption
    clean_caption = clean_caption.replace(REF_START_TAG, '').replace(REF_END_TAG, '')
    clean_caption = clean_caption.replace(REL_START_TAG, '').replace(REL_END_TAG, '')
    res = re.findall(grounded_pattern, grounded_caption)
                                                            
    last_tag = None
    last_tag_value = None
    for item in res:
        clean_item = re.sub(r'<.*?>', '', item)

        if item.startswith(BOX_START_TAG):
            clean_caption = clean_caption.replace(item, '')
            try:
                clean_item = json.loads(clean_item)
            except Exception as e:
                print('Invalid format:', clean_item)
                raise e
            if last_tag == REF_START_TAG:
                objects[last_tag_value].extend(clean_item)
            elif last_tag == REL_START_TAG:
                relations[last_tag_value].append(clean_item)
            else:
                objects['obj'].extend(clean_item)
        else:
            last_tag = REF_START_TAG if item.startswith(REF_START_TAG) else REL_START_TAG
            last_tag_value = clean_item  
    return objects

def varygrounding_2qwen(data, BOX_SCALE=999):
    # 提取对象
    objects = extract_obj(data["conversations"][1]['value'])
    image_h_w = [int(data["height"]),  int(data["width"])]
    # 将vary格式的bbxo还原
    restor_objects = od_restore_bbox(objects, image_h_w, BOX_SCALE = BOX_SCALE)
    if len(objects.items()) > 0:
        # 随机翻转变换，得到bbox_new
        # 再将bbox_new变换至vary格式
        data["conversations"][1]['value'] = dumy_obj(restor_objects)
    return data

def restore_bbox_in_json(data, BOX_SCALE=999):
    """
    从标注文本中提取并还原边界框为图像的实际尺寸。
    
    Args:
        data (dict): 包含图像及标注信息的字典。
        
        BOX_SCALE (int): 缩放比例,默认为999。
    Returns:
        dict: 更新后的数据，包含还原的边界框。
    """
    # 加载图像获取尺寸
    
    height, width = [data["height"],  data["width"]]
    
    # 定义恢复边界框的函数
    def restore_bbox(bboxes, height, width, BOX_SCALE):
        restored_bboxes = []
        for bbox in bboxes:
            # 按比例还原到原始大小
            bbox = [
                int(bbox[0]/BOX_SCALE * max(height, width)),
                int(bbox[1]/BOX_SCALE * max(height, width)),
                int(bbox[2]/BOX_SCALE * max(height, width)),
                int(bbox[3]/BOX_SCALE * max(height, width)),
            ]
            if height == width:
                pass
            elif height < width:
                delta=(width-height)//2
                bbox[1]-= delta
                bbox[3]-= delta
            else:
                delta=(height - width)// 2
                bbox[0]-= delta
                bbox[2]-= delta
            for i in range(len(bbox)):
                if bbox[i]< 0:
                    bbox[i]= 0
            # 归一化到1000
            bbox = [int(coord / dim * 999) for coord, dim in zip(bbox, [width, height, width, height])]
            
            restored_bboxes.append(bbox)
        return restored_bboxes

    # 提取原始标注框
    for conv in data['conversations']:
        if '<box>' in conv['value']:
            # 提取出归一化的边界框
            bbox_match = re.findall(r'<box>\[(.*?)\]</box>', conv['value'])
            if bbox_match:
                bbox_data = json.loads(f"[{bbox_match[0]}]")  # 转换为列表
                # 恢复边界框
                restored_bboxes = restore_bbox(bbox_data, height, width, BOX_SCALE)
                # 更新数据中的边界框
                restored_bboxes_str = json.dumps(restored_bboxes)
                conv['value'] = re.sub(r'<box>.*?</box>', f'<box>{restored_bboxes_str}</box>', conv['value'])

    return data

# 提取问答对
def extract_qa_pairs(data):
    qa_pairs = []
    data = restore_bbox_in_json(data)

    image_path = data["image"]  # 将相对路径转换为绝对路径
    conversations = data["conversations"]
    
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

def process_dataset(sourdir, output_dir, split_ratio=0.8):
    # 获取sourdir下的所有子文件夹
    subdirs = [d for d in os.listdir(sourdir) if os.path.isdir(os.path.join(sourdir, d))]

    for subdir in subdirs:
        subdir_path = os.path.join(sourdir, subdir)
        label_dir = subdir_path
        img_dir = subdir_path

        train_data = []
        val_data = []
        all_data = []

        # 获取所有标注文件
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]

        random.shuffle(label_files)  # 打乱文件顺序

        # 处理训练集文件
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            try:
                with open(label_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                    # 确保图像路径是绝对路径
                    image_name = os.path.basename(data["image"])
                    image_path = os.path.join(img_dir, image_name)
                    data["image"] = image_path  # 更新图像路径为绝对路径

                    if is_grounding_task(data):
                        if valid_grounding(label_path):
                            converted_data = convert_vary_2grounding(data)
                    else:
                        converted_data = extract_qa_pairs(data)

                    all_data.extend(converted_data)
            except Exception as e:
                print(f"处理文件 {label_path} 时发生错误: {e}")

        # 划分训练集和验证集
        train_size = int(len(all_data) * split_ratio)  # 计算训练集大小
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]

        # 保存训练集数据
        train_output_path = os.path.join(output_dir, f'{subdir}_train.jsonl')
        if train_data:
            save_data_as_jsonl(train_data, train_output_path)
            print(f"已保存训练集数据到 {train_output_path}")

        # 保存验证集数据
        val_output_path = os.path.join(output_dir, f'{subdir}_val.jsonl')
        if val_data:
            save_data_as_jsonl(val_data, val_output_path)
            print(f"已保存验证集数据到 {val_output_path}")


def is_grounding_task(data):
    conversations = data['conversations']
    for convo in conversations:
        if convo["from"] == "gpt":
            if re.search(r"<ref>.*</ref>", convo["value"]) and re.search(r"<box>.*</box>", convo["value"]):
                return True
    return False


# 转换为grounding任务格式
def convert_vary_2grounding(data):
    data = varygrounding_2qwen(data)
    target_data = {
        "query": "找到 <ref-object>",
        "response": "<bbox>",
        "images": [],
        "objects": ""
    }

    # 获取图像路径
    image_path = data["image"]
    # image_info = {
    #     "image": image_path  # 已经是绝对路径
    # }

    target_data["images"].append(image_path)
    target_data["objects"] = []
    # 获取conversation中的信息
    for conversation in data["conversations"]:
        if conversation["from"] == "gpt":
            ref_object = None
            bbox_info = None

            # 提取ref和box内容
            value = conversation["value"]
            if "<ref>" in value:
                ref_object = value.split("<ref>")[1].split("</ref>")[0]
            if "<box>" in value:
                bbox_info = value.split("<box>")[1].split("</box>")[0]

            if ref_object and bbox_info:
                # 处理多个bbox的情况
                bbox_list = process_bbox_info(bbox_info)

                # 创建object对象并添加到target_data["objects"]
                object_info = {
                    "caption": ref_object,
                    "bbox": bbox_list,  # 这个bbox是一个包含多个框的列表，如[[14, 235, 264, 351], [14, 33, 44, 55]]
                    "bbox_type": "real",
                    "image": 0  # 假设图片索引为0，依据实际需求调整
                }

                # 将object信息转为JSON格式字符串，并添加到目标数据中
                # target_data["objects"] += json.dumps([object_info], ensure_ascii=False, separators=(',', ':'))
                target_data["objects"].append(object_info) 

    return [target_data]


def process_bbox_info(bbox_info):
    """
    处理边界框信息，支持处理包含多个框的情况。
    如果bbox_info是字符串, 则清洗并解析。
    如果已经是列表，则直接返回。
    """
    if isinstance(bbox_info, str):
        try:
            bbox_list = json.loads(f"[{bbox_info}]")
        except json.JSONDecodeError as e:
            print(f"Error decoding bbox: {bbox_info} - {e}")
            bbox_list = []  # 解析失败时返回空列表
    elif isinstance(bbox_info, list):
        bbox_list = bbox_info
    else:
        print(f"Unsupported bbox_info format: {type(bbox_info)}")
        bbox_list = []

    return bbox_list[0]


def save_data_as_jsonl(data, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')

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

# 测试数据
sourdir = 'D:\code\data_extract\save\save/'
output_dir = 'D:\code\data_extract\save/new'  # 设定输出路径
split_ratio = 0.8  # 80% 训练集，20% 验证集

# 执行数据处理
process_dataset(sourdir, output_dir, split_ratio)
