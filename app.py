import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device
import track
import supervision as sv

def process_webcam(model, config=dict(conf=0.1, iou=0.45, classes=None), counting_zone=None, show_labels=True):
    cap = cv2.VideoCapture('street_5.mp4')
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 시작할 수 없습니다.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 비디오의 FPS
    slow_motion_factor = 0.3  # 느린 속도 배수
    slow_fps = int(original_fps * slow_motion_factor)  # 조정된 FPS

    video_info = sv.VideoInfo(fps=slow_fps,
                              width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              total_frames=0)

    model, _ = track.setup_model_and_video_info(model, config, None)
    byte_tracker = track.create_byte_tracker(video_info)
    annotators_list, trace_annotator, label_annotator, dot_annotator = track.setup_annotators()
    polygon_zone, polygon_zone_annotator = track.setup_counting_zone(counting_zone, video_info) if counting_zone else (None, None)

    # 영상 저장을 위한 VideoWriter 객체 초기화 (MP4 형식, 조정된 FPS로)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱 지정
    # out = cv2.VideoWriter('output_slow_motion.mp4', fourcc, slow_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def callback(frame, index):
        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=608, augment=False)
        detections = track.ExtendedDetections.from_yolov9(results)
        return track.annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model, dot_annotator)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = callback(frame, None)
        
        # 처리된 프레임을 표시하고 저장
        cv2.imshow('Processed Webcam Feed', annotated_frame)
        # out.write(annotated_frame)  # 처리된 프레임을 파일에 씀
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()  # 영상 저장 완료
    cv2.destroyAllWindows()

# 모델 로딩 및 설정
model_path = 'best_555.pt'
device = select_device('cpu')
model = DetectMultiBackend(model_path, device=device, dnn=False)
model = AutoShape(model)

# 웹캠 프로세스 시작
process_webcam(model)
