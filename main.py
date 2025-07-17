import cv2
import torch
import numpy as np
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
import supervision as sv
from shapely.geometry import Point, Polygon

region_colors = [
    (255, 0, 255),
    (0, 255, 255),
    (86, 0, 254),
    (0, 128, 255),
    (235, 183, 0),
    (255, 34, 134),
]

class MultipleObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cfg_regions = self.CFG["regions"] if "regions" in self.CFG else self.CFG.get("region", [])
        self.regions = cfg_regions

        # Prepare separate counters and sets for each region
        self.in_counts = [0] * len(self.regions)
        self.out_counts = [0] * len(self.regions)
        self.counted_ids = [set() for _ in range(len(self.regions))]

        self.region_initialized = False

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]

    def initialize_region_geometry(self):
        self.Point = Point
        self.Polygon = Polygon

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
            if region_width < region_height and current_centroid[0] > prev_position[0]:
                going_in = True
            elif region_width >= region_height and current_centroid[1] > prev_position[1]:
                going_in = True

            if going_in:
                self.in_counts[region_idx] += 1
            else:
                self.out_counts[region_idx] += 1

            self.counted_ids[region_idx].add(track_id)

    def display_counts(self, plot_im):
        for i, region_points in enumerate(self.regions):
            xs = [pt[0] for pt in region_points]
            ys = [pt[1] for pt in region_points]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))

            text_str = f"{self.in_counts[i] + self.out_counts[i]}"
            cv2.putText(
                plot_im,
                text_str,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                5
            )

    def process(self, frame):
        if not self.region_initialized:
            self.initialize_region_geometry()
            self.region_initialized = True

        # Extract object tracks
        self.extract_tracks(frame)

        # Initialize annotator
        self.annotator = SolutionAnnotator(frame, line_width=self.line_width)

        # Draw each region boundary
        for idx, region_points in enumerate(self.regions):
            color = region_colors[idx]
            self.annotator.draw_region(
                reg_pts=region_points,
                color=color,
                thickness=self.line_width * 2
            )
            b, g, r = color
            frame = sv.draw_filled_polygon(
                scene = frame,
                Polygon = np.array(region_points),
                color = sv.Color(r=r, g=g, b=b),
                opacity = 0.25
            )

        # For each detected object, update counting logic for each region
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Draw bounding box
            self.annotator.box_label(box, label=self.names[cls])
            # Update track history
            self.store_tracking_history(track_id, box)

            current_centroid = (
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            )

            # Previous position from the last frame
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]

            # Now attemp to count crossing for each region
            for r_idx, region_points in enumerate(self.regions):
                self.count_objects_in_region(
                    region_idx=r_idx,
                    region_points=region_points,
                    current_centroid=current_centroid,
                    track_id=track_id,
                    prev_position=prev_position,
                    cls=cls
                )

            plot_im = self.annotator.result()

            # Display the counts
            self.display_counts(plot_im)

            # Display output with base class function
            self.display_output(plot_im)

            # Return results
            return SolutionResults(
                plot_im=plot_im,
                total_tracks=len(self.track_ids)
            )
        
if __name__ == '__main__':
    # Khởi tạo video capture với file video
    video_path = "./raw_data/cars_test.mp4"  # Cập nhật đường dẫn
    cap = cv2.VideoCapture(video_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("output1.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
    counter = MultipleObjectCounter(
        show=True,
        region=region_points,
        model="best1024.pt",
        classes=["cars"]
    )
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        
        annotated_frame = counter.process(frame)
        results = counter(frame)
        frame = results.plot_im
        video_writer.write(frame)

        # Display frame (optional, comment out if not needed)
        cv2.imshow("YOLOv11 Object Counting", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to output.mp4")