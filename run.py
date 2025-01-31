import cv2
from ultralytics import YOLO
import numpy as np
import deep_sort_oh
from deep_sort_oh.deepsort_tracker import DeepSort

video_name = "./data/video_initial_raw.mov"
output_video_name = "output_deepsort_oh.mp4"

# deepsot算法
# 初始化YOLOv11模型
model = YOLO('./model/yolo11n.pt')

# 初始化DeepSORT
# 使用clip模型
# deepsort = DeepSort(max_age=30, max_iou_distance=0.5, nms_max_overlap=1, embedder='clip_ViT-B/16', embedder_wts="./model/ViT-B-16.pt")
# 使用torchreid模型
# deepsort = DeepSort(max_age=30, max_iou_distance=0.8, nms_max_overlap=0.8, embedder='torchreid')
# 无模型
deepsort = DeepSort(max_age=100, nms_max_overlap=0.8, n_init=3)

# 加载视频文件
cap = cv2.VideoCapture(video_name)

# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取输入视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义输出视频的编码方式和输出文件路径
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式(XVID-avi)
out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

sum = 0. # 帧数
total_sum = 0 # 总人数
tracked_ids = set()  # 用于记录所有被追踪的 ID
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用YOLOv11检测目标
    results = model(frame)  # 输入一帧图像
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取边界框数据 (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.cpu().numpy()  # 获取检测的置信度
    class_ids = results[0].boxes.cls.cpu().numpy()  # 获取分类ID

    # DeepSORT输入需要的格式是 [x1, y1, w, h] 和 [confidence]
    detections = []
    for i in range(len(boxes)):
        if confidences[i] > 0.25 and class_ids[i] == 0:  # 设置置信度阈值 并且 要是人的类别
            x1, y1, x2, y2 = boxes[i]
            detection = ([x1, y1, x2-x1, y2-y1], confidences[i])
            detections.append(detection)
    
    sum += 1    
    print(sum)

    # 使用DeepSORT进行追踪
    # 如果 detections 不为空，则使用 DeepSORT 进行追踪
    if len(detections) > 0:
        try:
            tracks = deepsort.update_tracks(detections, frame=frame)
            gts = []
            color1 = (0, 255, 0)  # 追踪框颜色
            color2 = (255, 0, 0)  # ID颜色
            fontsize = 1.5  # 0.5
            fontline = 3  # 2
            
            # 绘制追踪框
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()  # 获取目标的坐标
                track_id = track.track_id  # 获取追踪ID

                # 绘制框和ID
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color1, 2)
                cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color2, fontline)

                # 统计当前帧中的人数，添加新 ID 到 tracked_ids
                if track_id not in tracked_ids:
                    tracked_ids.add(track_id)
                    total_sum += 1
            
            # 统计当前帧中的人数
            active_tracks = [track for track in tracks if track.is_confirmed() and track.time_since_update <= 1]
            person_count = len(active_tracks)

            # 在左上角显示人数
            cv2.putText(frame, f'People Count: {person_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color2, fontline)
            cv2.putText(frame, f'Total People: {total_sum}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color2, fontline)

        except Exception as e:
            print(f"Error during tracking: {e}")
            continue

    # 将帧写入输出视频
    out.write(frame)

cap.release()
out.release()



# 显示视频的前几帧
import matplotlib.pyplot as plt

video_path = './' + output_video_name

cap = cv2.VideoCapture(video_path)

sample = 0 
while sample < 4:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    plt.imshow(frame)
    plt.axis('off')  
    plt.show()
    
    plt.pause(0.01)
    sample += 1

cap.release()