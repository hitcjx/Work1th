import csv
import json

def read_labels(csv_file):
    """读取 CSV 标注文件，返回关键帧信息列表。
    """
    key_frames = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            # 跳过前 6 行注释
            for _ in range(6):
                next(reader)

            for row_index, row in enumerate(reader, start=7):
                try:
                    # 检查行数据长度是否足够
                    if len(row) < 6:
                        print(f"Error: Row {row_index} has insufficient data: {row}")
                        continue

                    # 解析数据
                    video_name = json.loads(row[1])[0]
                    action_completed = bool(int(row[2]))
                    time = float(json.loads(row[3])[0]) if json.loads(row[3]) else None
                    error_type_str = json.loads(row[5]).get("1", "default")  # 使用 get() 获取值，避免 KeyError

                    # 转换错误类型
                    error_type = {
                        "None": 0,
                        "Heel Raise": 1,
                        "Lean Forward": 2,
                        "Knee Over Toe": 3,
                        "Knee Valgus": 4,
                        "Hip Imbalance": 5,
                        "Squat Not Low Enough": 6,
                    }.get(error_type_str, 0)

                    # 处理时间
                    time_ms = int(time * 1000) if time is not None else None

                    key_frames.append({
                        'file_id': video_name,
                        'time': time_ms,
                        'action_completed': action_completed,
                        'error_type': error_type
                    })

                except (IndexError, json.JSONDecodeError, ValueError) as e:
                    print(f"Error processing row {row_index}: {row}. Error: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    return key_frames

# 调用函数并打印结果
csv_file = '../data/test.csv'  # 将 'your_csv_file.csv' 替换为你的 CSV 文件路径
key_frames = read_labels(csv_file)
if key_frames:
    for key_frame in key_frames:
        print(key_frame)