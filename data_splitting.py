import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục chứa dữ liệu
input_dir = "output_folder"

# Thư mục đích để lưu dữ liệu đã chia
dataset_dir = "dataset"

# Tỷ lệ chia dữ liệu (80% train, 20% valid)
train_ratio = 0.8

# Các đuôi file ảnh phổ biến
image_extensions = [".jpg", ".jpeg", ".png"]

# Tạo file classes.txt
classes = ["car"]  # Một lớp duy nhất
classes_file = os.path.join(dataset_dir, "classes.txt")
os.makedirs(dataset_dir, exist_ok=True)
with open(classes_file, "w", encoding="utf-8") as f:
    for cls in classes:
        f.write(cls + "\n")
print(f"Đã tạo file classes.txt tại {classes_file}")

# Tạo cấu trúc thư mục dataset
os.makedirs(os.path.join(dataset_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "valid", "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "valid", "labels"), exist_ok=True)

# Liệt kê tất cả các file .txt và tìm ảnh tương ứng
pairs = []
for file_name in os.listdir(input_dir):
    if file_name.endswith(".txt"):
        base_name = os.path.splitext(file_name)[0]
        for ext in image_extensions:
            image_path = os.path.join(input_dir, base_name + ext)
            if os.path.exists(image_path):
                pairs.append((base_name + ext, file_name))
                break
        else:
            print(f"Không tìm thấy ảnh tương ứng cho: {file_name}")

# Kiểm tra file nhãn để đảm bảo class_id hợp lệ
def check_labels(labels_dir, max_class_id=0):
    for txt_file in os.listdir(labels_dir):
        if txt_file.endswith(".txt"):
            try:
                with open(os.path.join(labels_dir, txt_file), "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, 1):
                        if not line.strip():
                            continue  # Bỏ qua dòng trống
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Lỗi định dạng trong {txt_file}, dòng {line_number}: {line.strip()}")
                            continue
                        try:
                            class_id = int(float(parts[0]))
                            x, y, w, h = map(float, parts[1:])
                            if class_id > max_class_id:
                                print(f"Lỗi: class_id {class_id} trong {txt_file}, dòng {line_number} vượt quá số lớp (0)")
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                print(f"Lỗi: Giá trị ngoài phạm vi [0, 1] trong {txt_file}, dòng {line_number}: {line.strip()}")
                        except ValueError:
                            print(f"Lỗi định dạng số trong {txt_file}, dòng {line_number}: {line.strip()}")
            except UnicodeDecodeError:
                print(f"Lỗi mã hóa trong {txt_file}. Thử đọc với encoding khác (ví dụ: latin1).")
                # Thử đọc lại với encoding khác
                with open(os.path.join(labels_dir, txt_file), "r", encoding="latin1") as f:
                    for line_number, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Lỗi định dạng trong {txt_file}, dòng {line_number}: {line.strip()}")
                            continue
                        try:
                            class_id = int(float(parts[0]))
                            if class_id > max_class_id:
                                print(f"Lỗi: class_id {class_id} trong {txt_file}, dòng {line_number} vượt quá số lớp (0)")
                        except ValueError:
                            print(f"Lỗi định dạng số trong {txt_file}, dòng {line_number}: {line.strip()}")

# Kiểm tra nhãn trước khi chia
temp_labels_dir = input_dir
check_labels(temp_labels_dir)

# Chia dữ liệu thành train và valid
if len(pairs) == 0:
    print("Không tìm thấy cặp ảnh-nhãn nào!")
else:
    # Chia ngẫu nhiên
    train_pairs, valid_pairs = train_test_split(pairs, train_size=train_ratio, random_state=42)

    # Sao chép file vào thư mục train
    for image_name, txt_name in train_pairs:
        # Sao chép ảnh
        shutil.copy(
            os.path.join(input_dir, image_name),
            os.path.join(dataset_dir, "train", "images", image_name)
        )
        # Sao chép nhãn
        shutil.copy(
            os.path.join(input_dir, txt_name),
            os.path.join(dataset_dir, "train", "labels", txt_name)
        )
        print(f"Đã sao chép (train): {image_name}, {txt_name}")

    # Sao chép file vào thư mục valid
    for image_name, txt_name in valid_pairs:
        # Sao chép ảnh
        shutil.copy(
            os.path.join(input_dir, image_name),
            os.path.join(dataset_dir, "valid", "images", image_name)
        )
        # Sao chép nhãn
        shutil.copy(
            os.path.join(input_dir, txt_name),
            os.path.join(dataset_dir, "valid", "labels", txt_name)
        )
        print(f"Đã sao chép (valid): {image_name}, {txt_name}")

    # Kiểm tra nhãn trong thư mục train và valid
    check_labels(os.path.join(dataset_dir, "train", "labels"))
    check_labels(os.path.join(dataset_dir, "valid", "labels"))

print(f"Hoàn tất! Tổng cộng {len(train_pairs)} file trong train, {len(valid_pairs)} file trong valid.")