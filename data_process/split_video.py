import cv2
import os

# Đường dẫn video
video_path = "./raw_data/cars_test.mp4"

# Thư mục lưu ảnh
output_dir = "./data/cars_test"
os.makedirs(output_dir, exist_ok=True)

# Mở video
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Không mở được video!"

frame_rate = 5  # số frame/giây muốn lấy (tùy chỉnh)

count = 0
saved_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Lấy frame theo frame_rate
    if int(count % round(fps // frame_rate)) == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    count += 1

cap.release()
print(f"Đã lưu {saved_count} ảnh vào {output_dir}")