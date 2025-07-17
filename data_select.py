import os
import shutil

# Đường dẫn đến thư mục data
data_dir = "data"

# Thư mục đích để lưu các file .txt và ảnh tương ứng
output_dir = "output_folder"

# Các đuôi file ảnh phổ biến
image_extensions = [".jpg", ".jpeg", ".png"]

# Tạo thư mục đích nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Duyệt qua các thư mục con (cars1, cars2, ...)
subfolders = ["cars1", "cars2", "cars3", "cars4", "cars5"]

for subfolder in subfolders:
    subfolder_path = os.path.join(data_dir, subfolder)
    
    # Kiểm tra xem thư mục con có tồn tại không
    if not os.path.exists(subfolder_path):
        print(f"Thư mục {subfolder_path} không tồn tại!")
        continue
    
    # Duyệt qua tất cả các file trong thư mục con
    for file_name in os.listdir(subfolder_path):
        # Chỉ xử lý các file .txt
        if file_name.endswith(".txt"):
            txt_path = os.path.join(subfolder_path, file_name)
            base_name = os.path.splitext(file_name)[0]  # Lấy tên file không có đuôi
            
            # Tạo tên file mới với tiền tố từ thư mục con
            new_txt_name = f"{subfolder}_{file_name}"
            new_txt_path = os.path.join(output_dir, new_txt_name)
            
            # Tìm file ảnh tương ứng
            image_found = False
            for ext in image_extensions:
                image_path = os.path.join(subfolder_path, base_name + ext)
                if os.path.exists(image_path):
                    image_found = True
                    new_image_name = f"{subfolder}_{base_name + ext}"
                    new_image_path = os.path.join(output_dir, new_image_name)
                    
                    # Sao chép file .txt và file ảnh
                    shutil.copy(txt_path, new_txt_path)
                    shutil.copy(image_path, new_image_path)
                    print(f"Đã sao chép: {new_txt_name} và {new_image_name}")
                    break
            
            if not image_found:
                print(f"Không tìm thấy ảnh tương ứng cho: {file_name} trong {subfolder}")

print("Hoàn tất!")