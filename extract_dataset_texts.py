import os
import json


def generate_category_json(target_dirs):
    """
    遍历指定目录，查找COCO格式json，生成对应的类别文本json。
    """
    processed_count = 0

    for target_root in target_dirs:
        if not os.path.exists(target_root):
            print(f"[Warning] 路径不存在，跳过: {target_root}")
            continue

        print(f"-->正在扫描目录: {target_root}")

        # os.walk 递归遍历所有子文件夹
        for dirpath, dirnames, filenames in os.walk(target_root):
            for filename in filenames:

                # 过滤条件1: 必须是json文件
                if not filename.endswith(".json"):
                    continue

                # 过滤条件2: 排除掉我们自己生成的 categories json，防止死循环或重复处理
                if "_categories.json" in filename:
                    continue

                full_path = os.path.join(dirpath, filename)

                try:
                    # 尝试读取 JSON
                    with open(full_path, 'r', encoding='utf-8') as f:
                        # 为了防止读取超大文件卡死，先读一部分判断是否像 COCO (可选，这里直接读)
                        data = json.load(f)

                    # 判定条件: 必须包含 'categories' 且包含 'images'，才是标注文件
                    if isinstance(data, dict) and 'categories' in data and 'images' in data:

                        # -------------------------------------------------
                        # 核心逻辑：提取并排序
                        # -------------------------------------------------
                        cats = data['categories']
                        # 务必按 id 排序，否则 mmdet 训练时的 id 映射会乱
                        cats.sort(key=lambda x: x['id'])

                        class_names = [c['name'] for c in cats]

                        # -------------------------------------------------
                        # 生成文件名: 子文件夹名_categories.json
                        # -------------------------------------------------
                        sub_folder_name = os.path.basename(dirpath)
                        # 如果 json 文件就在根目录下，可能没有子目录名，兜底使用 filename
                        if not sub_folder_name:
                            sub_folder_name = os.path.splitext(filename)[0]

                        output_filename = f"{sub_folder_name}_categories.json"
                        output_path = os.path.join(dirpath, output_filename)

                        # 写入新的类别文本文件
                        with open(output_path, 'w', encoding='utf-8') as out_f:
                            json.dump(class_names, out_f, ensure_ascii=False, indent=2)

                        print(f"[处理成功] 源文件: {filename}")
                        print(f"           类别数: {len(class_names)}")
                        print(f"           生成至: {output_path}")
                        processed_count += 1

                except Exception as e:
                    # 遇到非标准 json 或损坏文件跳过
                    print(f"[Error] 处理 {filename} 时出错: {e}")

    print(f"\n========================================")
    print(f"全部完成! 共生成了 {processed_count} 个类别文本文件。")
    print(f"========================================")


if __name__ == "__main__":
    # 在这里配置你的数据集根目录路径
    # 假设你的 LAE-1M 数据放在 data/LAE-1M/ 下
    dataset_roots = [
        "data/LAE-1M/LAE-COD",
        "data/LAE-1M/LAE-FOD",
        # 如果有检测集文件夹，也可以加在这里，例如:
        # "data/LAE-1M/Detection"
    ]

    generate_category_json(dataset_roots)
