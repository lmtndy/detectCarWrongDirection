from ultralytics import SAM
import cv2
model = SAM("sam2.1_b.pt")

# Load a model
input_image = r"D:\Study\IMAGE\ultralytics\car_wrong_direction\z6709307257678_461058d484697e4e6760f698b218bb61.jpg"
results = model(input_image, 
                points=[[408, 256]])

for i, res in enumerate(results):
    normalized_bboxes = res.boxes.xywhn

    # Tạo tên file đầu ra (đổi đuôi từ .jpg sang .txt)
    output_txt = input_image.replace(".jpg", ".txt")

    with open(output_txt, "w", encoding="utf-8") as f:
        for nbbox in normalized_bboxes:
            x, y, w, h = nbbox.tolist()  # Chuyển tensor sang list để format an toàn
            f.write("0 {} {} {} {}\n".format(x, y, w, h))

image = results[0].plot(labels=False)
cv2.imshow("test", image)
cv2.waitKey(0)