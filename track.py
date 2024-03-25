import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device
from utils.general import set_logging
import supervision as sv
from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD
from itertools import combinations

# ExtendedDetections 클래스 정의
class ExtendedDetections(BaseDetections):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids = []

    @classmethod
    def from_yolov9(cls, yolov9_results) -> 'ExtendedDetections':
        xyxy, confidences, class_ids, centroids = [], [], [], []
        for det in yolov9_results.pred:
            for *xyxy_coords, conf, cls_id in reversed(det):
                xyxy.append(torch.stack(xyxy_coords).cpu().numpy())
                confidences.append(float(conf))
                class_ids.append(int(cls_id))
                center = ((xyxy_coords[0] + xyxy_coords[2]) / 2, (xyxy_coords[1] + xyxy_coords[3]) / 2)
                centroids.append(center)
                

        class_names = np.array([yolov9_results.names[i] for i in class_ids])
        if not xyxy:
            return cls.empty()

        detections = cls(
            xyxy=np.vstack(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            data={CLASS_NAME_DATA_FIELD: class_names},
        )
        detections.centroids = centroids
        return detections
    

set_logging(verbose=False)

device = select_device('cpu')
model = DetectMultiBackend(weights='best_555.pt', device=device, data='data/coco.yaml', fuse=True)
model = AutoShape(model)

def prepare_yolov9(model, conf=0.9, iou=0.7, classes=None, agnostic_nms=False, max_det=1000):
    model.conf = conf
    model.iou = iou
    model.classes = classes
    model.agnostic = agnostic_nms
    model.max_det = max_det
    return model

def create_byte_tracker(video_info):
    # Setup BYTETracker with video information
    return sv.ByteTrack(track_thresh=0.25, track_buffer=250, match_thresh=0.95, frame_rate=video_info.fps)

def setup_annotators():
    # Initialize various annotators for bounding boxes, traces, and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    round_box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    corner_annotator = sv.BoxCornerAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, color_lookup=sv.ColorLookup.TRACK)
    dot_annotator = sv.DotAnnotator()  # DotAnnotator 추가
    return [bounding_box_annotator, round_box_annotator, corner_annotator], trace_annotator, label_annotator, dot_annotator

def setup_counting_zone(counting_zone, video_info):
    # Configure counting zone based on provided parameters
    if counting_zone == 'whole_frame':
        polygon = np.array([[0, 0], [video_info.width-1, 0], [video_info.width-1, video_info.height-1], [0, video_info.height-1]])
    else:
        polygon = np.array(counting_zone)
    polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(video_info.width, video_info.height), triggering_position=sv.Position.CENTER)
    polygon_zone_annotator = sv.PolygonZoneAnnotator(polygon_zone, sv.Color.ROBOFLOW, thickness=2*(2 if counting_zone=='whole_frame' else 1), text_thickness=1, text_scale=0.5)
    return polygon_zone, polygon_zone_annotator

def setup_model_and_video_info(model, config, source_path):
    # Initialize and configure YOLOv9 model
    model = prepare_yolov9(model, **config)
    
    if source_path is None:
        # 웹캠 스트림을 처리하는 경우, 직접 VideoInfo 객체를 생성
        # 이 예에서는 대략적인 값을 사용합니다. 실제로는 웹캠의 속성을 조회하여 설정해야 합니다.
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        cap.release()  # VideoCapture 객체는 여기에서만 필요하므로 바로 해제
        
        video_info = sv.VideoInfo(fps=fps, width=width, height=height, total_frames=0)
    else:
        # 비디오 파일 경로가 제공된 경우, 기존 로직을 사용하여 VideoInfo 생성
        video_info = sv.VideoInfo.from_video_path(source_path)
    
    return model, video_info

# 거리 계산 함수
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 각도 계산 함수
def calculate_angle(point1, point2):
    # 두 점 사이의 벡터 계산
    vec = np.array(point2) - np.array(point1)
    # y축과의 각도 계산을 위해 y축 단위 벡터 정의
    y_axis = np.array([1, 0])
    # 벡터와 y축 사이의 코사인 각도 계산
    cos_theta = np.dot(vec, y_axis) / (np.linalg.norm(vec) * np.linalg.norm(y_axis))
    # 코사인 역함수를 사용하여 라디안으로 각도 계산
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    # 라디안을 도(degree)로 변환
    angle_deg = np.degrees(angle_rad)

    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    
    return angle_deg

last_positions = {}

# 프레임 주석 달기 함수
def annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model, dot_annotator):
    if index is None:
        # 실제 로직에 맞게 적절히 조정해야 합니다.
        index = 0
    section_index = int(index / (video_info.total_frames / len(annotators_list)) if video_info.total_frames else 1)
    detections = byte_tracker.update_with_detections(detections)
    annotated_frame = frame.copy()
    if counting_zone is not None:
        is_inside_polygon = polygon_zone.trigger(detections)
        detections = detections[is_inside_polygon]
        annotated_frame = polygon_zone_annotator.annotate(annotated_frame)

    # Annotate frame with traces
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    # Annotate frame with various bounding boxes
    annotated_frame = annotators_list[section_index].annotate(scene=annotated_frame, detections=detections)

    # # Optionally, add labels to the annotations
    if show_labels:
        annotated_frame = add_labels_to_frame(label_annotator, annotated_frame, detections, model)
    annotated_frame = dot_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    frame_rgb = frame[..., ::-1]
    results = model(frame_rgb, size=640, augment=False)
    detection = ExtendedDetections.from_yolov9(results)

    id_positions = {}

    threshold_ByteTrack = 100
    threshold_Angle = 70
    threshold_Btw = 200
    static_positions = []  # 정적 위치와 해당 ID를 저장할 리스트 초기화

    for det, id in zip(detections.xyxy, detections.tracker_id):
        # det는 [x1, y1, x2, y2] 형태의 배열입니다.
        # 중심 좌표 계산
        center_x = (det[0] + det[2]) / 2
        center_y = (det[1] + det[3]) / 2
        center_int = (int(center_x), int(center_y))

        # 각 id에 대한 현재 위치 저장
        id_positions[id] = center_int

        if id in last_positions:
            # 이전 위치와 현재 위치 사이의 거리 계산
            last_pos = last_positions[id]
            ByteTrack_dist = calculate_distance(center_int, last_pos)
            print(f"ID_{id}: {center_int}  ---  ByteTrack dist: {ByteTrack_dist}")

            if ByteTrack_dist < threshold_ByteTrack:
                # 거리가 임계값 미만이면, static_positions에 추가
                static_positions.append((id, center_int))
        else:
            # 새로운 객체로, 이전 위치 정보 없음
            print(f"ID_{id}, {center_int}, New Object")
        
        # 마지막 위치 정보 업데이트
        last_positions[id] = center_int

    for (id1, pos1), (id2, pos2) in combinations(static_positions, 2):
        Btw_dist = calculate_distance(pos1, pos2)  # 위치 정보 추출
        angle = calculate_angle(pos1, pos2)  # 위치 정보 추출
        print(f"(ID_{id1},ID_{id2})  ---  Btw dist: {Btw_dist:.2f}  /  Btw Ang: {angle:.2f} ⁰")
        if Btw_dist < threshold_Btw and angle > threshold_Angle:
            cv2.line(annotated_frame, pos1, pos2, (0, 0, 255), 2)  # 빨간색 선으로 이어주기
            # 선의 중간 지점 계산
            mid_point = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
            # 각도 값을 빨간 선 위에 표시
            cv2.putText(annotated_frame, f"{angle:.2f}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print("----------------------------------------------")

    return annotated_frame

def add_labels_to_frame(annotator, frame, detections, model):
    # 각 검출된 객체에 대해 tracker_id만을 라벨로 생성
    labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id]
    # 수정된 라벨 리스트를 사용하여 프레임에 주석 추가
    return annotator.annotate(scene=frame, detections=detections, labels=labels)