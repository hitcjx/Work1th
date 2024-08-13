import cv2

def check_video(video_file):
    """检查视频文件是否可以正常读取，并打印每一帧的信息。"""

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_file}")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Frame {frame_count}: Shape = {frame.shape}, dtype = {frame.dtype}, ret = {ret}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 将 video_file 替换为你要测试的视频文件的路径
    video_file = "/home/xuan/下载/video_20240812_195248.mp4"
    check_video(video_file)