import cv2
import mediapipe as mp
import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

mp_pose = mp.solutions.pose

class SquatDataset(Dataset):
    def __init__(self, csv_file, keypoints_folder):
        self.key_frames = self.read_labels(csv_file)
        self.keypoints_folder = keypoints_folder

    def __len__(self):
        return len(self.key_frames)

    def __getitem__(self, idx):
        key_frame = self.key_frames[idx]
        file_id = key_frame['file_id']
        time_ms = key_frame['time']

        # 读取关键点数据
        formatted_time_ms = f"{time_ms:.2f}" if time_ms is not None else "None"
        keypoints_file = f"{self.keypoints_folder}/{file_id}_{formatted_time_ms}.json"
        try:
            with open(keypoints_file, 'r') as f:
                keypoints_data = json.load(f)
                keypoints = keypoints_data['keypoints']
        except FileNotFoundError:
            print(f"Error: Keypoints file not found: {keypoints_file}")
            # 返回一个空的二维数组，避免 NoneType 错误
            keypoints = np.zeros((17, 3), dtype=np.float32)

        # 数据预处理 (Min-Max 归一化)
        keypoints = self.normalize_keypoints(keypoints)

        # 转换为 PyTorch 张量
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        action_completed = torch.tensor(key_frame['action_completed'],
                                        dtype=torch.float32)
        error_type = torch.tensor(key_frame['error_type'], dtype=torch.long)

        return keypoints, action_completed, error_type

    def read_labels(self, csv_file):
        """读取 CSV 标注文件，返回关键帧信息列表。"""
        key_frames = []
        try:
            with open(csv_file, 'r') as f:
                print(f"Opened CSV file: {csv_file}")
                reader = csv.reader(f)
                # 跳过前 10 行注释
                for _ in range(10):
                    line = next(reader)
                    print(f"Skipped line: {line}")

                for row_index, row in enumerate(reader, start=11):
                    print(f"Processing row {row_index}: {row}")
                    try:
                        # 检查行数据长度是否足够
                        if len(row) < 6:
                            print(
                                f"Error: Row {row_index} has insufficient data: {row}"
                            )
                            continue

                        # 解析数据
                        video_name = json.loads(row[1])[0]
                        action_completed = bool(int(row[2]))

                        # 处理 time 字段，捕获 IndexError
                        try:
                            time = float(
                                json.loads(row[3])[0]
                            ) if json.loads(row[3]) else None
                        except IndexError:
                            print(
                                f"Error: Empty time list in row {row_index}. Setting time to None."
                            )
                            time = None

                        error_type_str = json.loads(row[5]).get("1", "default")

                        # 转换错误类型
                        error_type = {
                            "default": 0,
                            "Heel Raise": 1,
                            "Lean Forward": 2,
                            "Knee Over Toe": 3,
                            "Knee Valgus": 4,
                            "Hip Imbalance": 5,
                            "Squat Not Low Enough": 6,
                        }.get(error_type_str, 0)

                        # 处理时间，移到 try 块外部
                        time_ms = int(time * 1000) if time is not None else None

                        # 添加关键帧信息到列表
                        key_frame = {
                            'file_id': video_name,
                            'time': time_ms,
                            'action_completed': action_completed,
                            'error_type': error_type
                        }
                        key_frames.append(key_frame)

                        print(f"Extracted keyframe: {key_frame}")

                    except (IndexError, json.JSONDecodeError, ValueError) as e:
                        print(f"Error processing row {row_index}: {row}. Error: {e}")
                        continue

                print(f"Keyframes: {key_frames}")
                return key_frames

        except FileNotFoundError:
            print(f"Error: File not found: {csv_file}")
            return []  # 返回空列表，避免 NoneType 错误
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []  # 返回空列表，避免 NoneType 错误

    def normalize_keypoints(self, keypoints):
        """使用 Min-Max 方法归一化关键点坐标."""
        keypoints = np.array(keypoints)
        x_min = np.min(keypoints[0::3])
        x_max = np.max(keypoints[0::3])
        y_min = np.min(keypoints[1::3])
        y_max = np.max(keypoints[1::3])

        # 归一化 x 和 y 坐标，避免除以零
        keypoints[0::3] = (keypoints[0::3] - x_min) / (x_max - x_min + 1e-8)  # 加上一个很小的值
        keypoints[1::3] = (keypoints[1::3] - y_min) / (y_max - y_min + 1e-8)  # 加上一个很小的值

        return keypoints

    def extract_keypoints_from_video(self, video_file, output_folder):
        """从视频中提取关键点并保存到 JSON 文件."""
        cap = cv2.VideoCapture(video_file)

        # 检查视频是否打开成功
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_file}")
            return

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 检查读取到的帧
                print(f"Frame {frame_count}: Shape = {frame.shape}, dtype = {frame.dtype}")

                # 将 BGR 转换为 RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Mediapipe 处理
                results = pose.process(image)

                # 将 RGB 转换为 BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 提取关键点
                try:
                    landmarks = results.pose_landmarks.landmark
                    keypoints = []
                    for landmark in landmarks:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])

                    # 获取当前帧的时间戳 (秒) 并格式化
                    time_s = "{:.2f}".format(
                        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    )  # 保留两位小数并转换为字符串

                    # 保存关键点数据到 JSON 文件
                    output_file = f"{output_folder}/{video_file.split('/')[-1].split('.')[0]}_{time_s}.json"
                    print(f"Saving keypoints to: {output_file}")  # 打印输出文件名
                    with open(output_file, 'w') as f:
                        json.dump({'keypoints': keypoints}, f)

                except Exception as e:
                    print(f"Error processing frame: {e}")

                frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


# 使用示例
csv_file = '../data/test.csv'
keypoints_folder = '../data/keypoints'  # 存储关键点数据的文件夹
video_folder = '../data/videos'  # 视频文件所在的文件夹

dataset = SquatDataset(csv_file, keypoints_folder)
for key_frame in dataset.key_frames:
    video_file = f"{video_folder}/{key_frame['file_id']}"
    dataset.extract_keypoints_from_video(video_file, keypoints_folder)

# 创建数据集和数据加载器
dataset = SquatDataset(csv_file, keypoints_folder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 迭代数据加载器
for batch_idx, (keypoints, action_completed,
                error_type) in enumerate(dataloader):
    # ... (处理每个批次的数据)
    print(
        f"Batch {batch_idx}: keypoints.shape={keypoints.shape}, action_completed={action_completed}, error_type={error_type}"
    )