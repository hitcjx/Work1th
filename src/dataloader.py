import cv2
import mediapipe as mp
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

mp_pose = mp.solutions.pose


class SquatDataset(Dataset):
    def __init__(self, json_files, keypoints_folder, transform=None):
        self.key_frames = self.read_labels(json_files)  # 调用 read_labels 读取 JSON 文件
        self.keypoints_folder = keypoints_folder
        self.transform = transform

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
            keypoints = np.zeros((12, 3), dtype=np.float32)

        # 数据预处理 (Min-Max 归一化)
        keypoints = self.normalize_keypoints(keypoints)

        # 数据增强
        if self.transform:
            keypoints = self.transform(keypoints)

        #转化为pytorch张量
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        action_stage = torch.tensor(key_frame['action_stage'], dtype=torch.long)
        error_type = torch.tensor(key_frame['error_type'], dtype=torch.long)
        valid_squat = torch.tensor(key_frame['valid_squat'], dtype=torch.float32)

        return keypoints, action_stage, error_type, valid_squat

    def read_labels(self, json_files):
        """读取 JSON 标注文件，返回关键帧信息列表。"""
        key_frames = []
        for json_file in json_files:  # 遍历所有 JSON 文件
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                    # 获取视频文件名和 ID
                    file_data = data['file'][list(data['file'].keys())[0]]
                    video_name = file_data['fname']
                    video_id = file_data['fid']  # 获取视频 ID

                    # 遍历 metadata，提取关键帧信息
                    for metadata_key, metadata_value in data['metadata'].items():
                        time = metadata_value['z'][0] if metadata_value['z'] else None  # 处理空列表的情况
                        time_ms = int(time * 1000) if time is not None else None
                        action_stage = int(metadata_value['av'].get('2', -1))  # 获取动作阶段，默认为 -1
                        error_type = int(metadata_value['av'].get('3', 0))  # 获取错误类型，默认为 0
                        valid_squat = bool(int(metadata_value['av'].get('4', 0)))  # 获取是否为有效深蹲，默认为 False

                        key_frame = {
                            'file_id': video_name,
                            'video_id': video_id,  # 添加 video_id
                            'time': time_ms,
                            'action_stage': action_stage,
                            'error_type': error_type,
                            'valid_squat': valid_squat
                        }
                        key_frames.append(key_frame)

            except FileNotFoundError:
                print(f"Error: File not found: {json_file}")
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")

        return key_frames

    def normalize_keypoints(self, keypoints):
        """使用 Min-Max 方法归一化关键点坐标."""
        keypoints = np.array(keypoints)

        # 分别对 x, y, z 坐标进行归一化
        for i in range(3):  # 遍历 x, y, z 三个维度
            min_val = np.min(keypoints[:, i])
            max_val = np.max(keypoints[:, i])
            # 归一化，避免除以零
            keypoints[:, i] = (keypoints[:, i] - min_val) / (max_val - min_val + 1e-8)

        return keypoints

    def extract_keypoints_from_video(self, video_file, output_folder, key_frames):
        """从视频中提取关键帧的关键点并保存到 JSON 文件."""
        cap = cv2.VideoCapture(video_file)

        # 检查视频是否打开成功
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_file}")
            return

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            for key_frame in key_frames:
                if key_frame['file_id'] == video_file.split('/')[-1]:
                    time_ms = key_frame['time']
                    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
                    ret, frame = cap.read()
                    if not ret:
                        print(
                            f"Error: Could not read frame at {time_ms}ms from video: {video_file}"
                        )
                        continue

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
                        # 选择需要的 12 个关键点，并提取三维坐标
                        for i in [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
                            landmark = landmarks[i]
                            keypoints.append([landmark.x, landmark.y, landmark.z])

                        # 保存关键点数据到 JSON 文件
                        formatted_time_ms = f"{time_ms:.2f}" if time_ms is not None else "None"
                        output_file = f"{output_folder}/{video_file.split('/')[-1].split('.')[0]}_{formatted_time_ms}.json"
                        print(f"Saving keypoints to: {output_file}")
                        with open(output_file, 'w') as f:
                            json.dump({'keypoints': keypoints}, f)

                    except Exception as e:
                        print(f"Error processing frame: {e}")

        cap.release()
        cv2.destroyAllWindows()


# 数据增强函数
def random_rotate(keypoints):
    """对关键点进行随机旋转."""
    angle = np.random.uniform(-10, 10)  # 随机旋转角度
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    for i in range(12):  # 修改循环次数为 12
        x, y = keypoints[i, 0], keypoints[i, 1]  # 使用正确的索引访问 x 和 y 坐标
        rotated = np.dot(R, np.array([x, y]))
        keypoints[i, 0], keypoints[i, 1] = rotated[0], rotated[1]  # 更新 x 和 y 坐标

    return keypoints


def random_scale(keypoints):
    """对关键点进行随机缩放."""
    scale = np.random.uniform(0.9, 1.1)  # 随机缩放比例
    keypoints *= scale  # 直接对所有坐标进行缩放
    return keypoints

def random_translate(keypoints):
    """对关键点进行随机平移."""
    x_translation = np.random.uniform(-0.1, 0.1)  # 随机水平平移距离
    y_translation = np.random.uniform(-0.1, 0.1)  # 随机垂直平移距离
    keypoints[:, 0] += x_translation  # 对所有 x 坐标进行平移
    keypoints[:, 1] += y_translation  # 对所有 y 坐标进行平移
    return keypoints

# 使用示例
json_file = [
    '../data/1.json','../data/2.json','../data/3.json','../data/4.json','../data/5.json','../data/6.json',
    '../data/7.json','../data/8.json','../data/9.json','../data/10.json','../data/11.json','../data/12.json',
    '../data/13.json','../data/14.json','../data/15.json','../data/16.json']
keypoints_folder = '../data/keypoints'  # 存储关键点数据的文件夹
video_folder = '../data/videos'  # 视频文件所在的文件夹

# 定义数据增强变换
transform = random_rotate  # 选择一个数据增强函数，或者组合多个函数

# 提取关键点
dataset = SquatDataset(json_file,
                       keypoints_folder,
                       transform=transform)  # 创建一个临时 dataset 对象，仅用于提取关键点
for key_frame in dataset.key_frames:
    video_file = f"{video_folder}/{key_frame['file_id']}"
    dataset.extract_keypoints_from_video(video_file, keypoints_folder,
                                         dataset.key_frames)

# 创建数据集和数据加载器
dataset = SquatDataset(json_file, keypoints_folder,
                       transform=transform)  # 重新创建 dataset 对象，用于训练
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 迭代数据加载器
for batch_idx, (keypoints, action_stage, error_type, valid_squat) in enumerate(dataloader):
    # ... (处理每个批次的数据)
    print(
        f"Batch {batch_idx}: keypoints.shape={keypoints.shape}, action_stage={action_stage}, error_type={error_type}, valid_squat={valid_squat}"
    )