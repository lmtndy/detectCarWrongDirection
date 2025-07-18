import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
from shapely.geometry import Point, Polygon

class MultipleObjectCounter:
    def __init__(self, model_path, regions=None, class_names=["car"], show=True, show_in=True, show_out=True, line_width=2):
        """
        Khởi tạo bộ đếm đối tượng cho nhiều vùng.

        Args:
            model_path (str): Đường dẫn đến mô hình YOLOv11 (.pt file).
            regions (list, optional): Danh sách các điểm vùng, mỗi vùng là [[x1,y1], [x2,y2], ...].
            class_names (list): Danh sách tên lớp cần phát hiện (mặc định: ["car"]).
            show (bool): Hiển thị cửa sổ xem trực tiếp.
            show_in (bool): Hiển thị số lượng vào vùng.
            show_out (bool): Hiển thị số lượng ra vùng.
            line_width (int): Độ dày đường vẽ vùng và hộp.
        """
        # Khởi tạo CFG từ các tham số
        self.CFG = {
            "region": regions if regions is not None else [],  # Gán regions hoặc danh sách rỗng
            "show_in": show_in,
            "show_out": show_out,
            "show": show,
            "line_width": line_width
        }

        # Lấy regions từ CFG
        cfg_regions = self.CFG["region"]
        if not cfg_regions:
            print("Cảnh báo: Không có vùng nào được định nghĩa. Sử dụng danh sách vùng mặc định rỗng.")
        self.regions = cfg_regions

        # Khởi tạo mô hình YOLO
        self.model = YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Khởi tạo các thuộc tính khác
        self.class_names = class_names
        self.in_counts = [0] * len(self.regions)
        self.out_counts = [0] * len(self.regions)
        self.counted_ids = [set() for _ in self.regions]
        self.track_history = {}
        self.region_initialized = False
        self.line_width = line_width
        self.show = show
        self.show_in = show_in
        self.show_out = show_out

        # Khởi tạo annotator từ Supervision
        self.box_annotator = sv.BoxAnnotator(thickness=line_width)

    def initialize_region_geometry(self):
        self.Point = Point
        self.Polygon = Polygon
        self.region_initialized = True

    def count_objects_in_region(self, region_idx, region_points, current_centroid, track_id, prev_position):
        if prev_position is None or track_id in self.counted_ids[region_idx]:
            return
        polygon = self.Polygon(region_points)
        if polygon.contains(self.Point(current_centroid)):
            xs = [pt[0] for pt in region_points]
            ys = [pt[1] for pt in region_points]
            region_width = max(xs) - min(xs)
            region_height = max(ys) - min(ys)

            going_in = False
            if region_width < region_height and current_centroid[1] > prev_position[1]:
                going_in = True
            elif region_width >= region_height and current_centroid[0] > prev_position[0]:
                going_in = True

            if going_in:
                self.in_counts[region_idx] += 1
            else:
                self.out_counts[region_idx] += 1

            self.counted_ids[region_idx].add(track_id)

    def store_tracking_history(self, track_id, box):
        centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(centroid)
        if len(self.track_history[track_id]) > 2:
            self.track_history[track_id] = self.track_history[track_id][-2:]

    def display_counts(self, frame):
        for i, region_points in enumerate(self.regions):
            xs = [pt[0] for pt in region_points]
            ys = [pt[1] for pt in region_points]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))

            text_str = f"region {i+1}: In={self.in_counts[i]}, Out={self.out_counts[i]}" if self.show_in and self.show_out else \
                       f"region {i+1}: Total={self.in_counts[i] + self.out_counts[i]}"

            cv2.putText(
                frame,
                text_str,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    def process(self, frame):
        if not self.region_initialized:
            self.initialize_region_geometry()

        results = self.model.track(frame, persist=True, classes=[self.class_names.index("car")], conf=0.5)
        detections = sv.Detections.from_ultralytics(results[0])
        boxes = detections.xyxy
        track_ids = detections.tracker_id if detections.tracker_id is not None else []
        class_ids = detections.class_id

        # Tạo nhãn
        labels = [f"{self.class_names[cid]} {tid}" for cid, tid in zip(class_ids, track_ids)]
        
        # Vẽ hộp giới hạn
        frame = self.box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        # Vẽ nhãn thủ công bằng cv2.putText
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),  # Vị trí ngay trên hộp
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Vẽ vùng
        for idx, region_points in enumerate(self.regions):
            color = region_colors[idx % len(region_colors)]
            frame = sv.draw_polygon(
                scene=frame,
                polygon=np.array(region_points),
                color=sv.Color.from_rgb_tuple(color),
                thickness=self.line_width * 2
            )
            frame = sv.draw_filled_polygon(
                scene=frame,
                polygon=np.array(region_points),
                color=sv.Color.from_rgb_tuple(color),
                opacity=0.25
            )

        # Đếm đối tượng
        for box, track_id, cls in zip(boxes, track_ids, class_ids):
            if cls != self.class_names.index("car"):
                continue
            self.store_tracking_history(track_id, box)
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

            for r_idx, region_points in enumerate(self.regions):
                self.count_objects_in_region(
                    region_idx=r_idx,
                    region_points=region_points,
                    current_centroid=current_centroid,
                    track_id=track_id,
                    prev_position=prev_position
                )

        self.display_counts(frame)
        return frame

if __name__ == '__main__':
    region_colors = [
        (0, 255, 0),
        (0, 255, 255),
        (128, 0, 255),  
        (86, 0, 254),   
        (0, 204, 153),  
        (0, 128, 255),  

        (255, 0, 0),    
        (255, 34, 134), 
        (255, 102, 0),  
        (255, 255, 0),  
        (235, 183, 0),  
        (255, 51, 51),  
        (204, 102, 0),  
        (255, 153, 102) 
    ]
    # Định nghĩa các vùng
    region_points = [
        [[890, 14], [889, 65], [410, 65], [409, 14]],
        [[892, 70], [893, 119], [410, 114], [409, 68]],
        [[897, 126], [898, 164], [410, 162], [410, 122]],
        [[903, 168], [904, 214], [410, 208], [409, 167]],
        [[908, 216], [906, 264], [410, 259], [410, 214]],
        [[908, 265], [908, 313], [407, 311], [410, 264]],
        [[911, 323], [911, 373], [406, 370], [407, 321]],
        [[909, 375], [909, 421], [407, 416], [407, 375]],
        [[911, 424], [912, 477], [406, 473], [406, 418]],
        [[919, 478], [922, 524], [406, 518], [406, 477]],
        [[919, 529], [922, 582], [410, 577], [407, 523]],
        [[925, 583], [925, 624], [409, 615], [410, 580]],
        [[927, 628], [928, 669], [409, 664], [409, 624]],
        [[933, 671], [933, 713], [410, 712], [410, 667]]
    ]

    # Khởi tạo video capture với file video
    video_path = "D:/Study/image_ultralytics_learning/ultralytics/raw_data/cars_test.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file video {video_path}")
        exit()

    # Lấy thông số video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Khởi tạo video writer
    video_writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # Khởi tạo bộ đếm
    counter = MultipleObjectCounter(
        model_path="best1024.pt",
        regions=region_points,
        class_names=["car"],
        show=True,
        show_in=True,
        show_out=True,
        line_width=2
    )

    # Xử lý video
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Xử lý khung hình
        annotated_frame = counter.process(frame)
        video_writer.write(annotated_frame)

        # Hiển thị khung hình nếu show=True
        if counter.show:
            cv2.imshow("YOLOv11 Object Counting", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Giải phóng tài nguyên
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("Xử lý hoàn tất. Đầu ra được lưu tại output.mp4")