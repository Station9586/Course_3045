import cv2
import numpy as np
import tensorflow as tf
import os
from keras.api import preprocessing
from keras.api import applications
from keras.api.models import load_model
# 新增導入 Matplotlib
import matplotlib.pyplot as plt

def detect_face_mask(image_path,
                     face_net_proto="../models/deploy.prototxt.txt",
                     face_net_weights="../models/res10_300x300_ssd_iter_140000.caffemodel",
                     mask_model_path="../models/model-facemask.h5",
                     confidence_threshold=0.5):
    # 檢查模型檔案是否存在
    if not os.path.exists(face_net_proto) or not os.path.exists(face_net_weights):
        print(f"錯誤：找不到臉部偵測模型檔案。請確認路徑：\n{face_net_proto}\n{face_net_weights}")
        return
    if not os.path.exists(mask_model_path):
        print(f"錯誤：找不到口罩分類模型檔案。請確認路徑：{mask_model_path}")
        return

    # 載入臉部偵測模型 (OpenCV DNN)
    face_net = cv2.dnn.readNet(face_net_proto, face_net_weights)

    # 載入口罩分類模型 (TensorFlow/Keras)
    mask_model = load_model(mask_model_path)

    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return
    # 為了 Matplotlib 顯示，先複製一份原始圖片
    orig_for_display = image.copy()
    (h, w) = image.shape[:2]

    # --- 臉部偵測 ---
    # 將圖片轉為 blob (調整大小、均值減去)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 將 blob 輸入神經網路
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []       # 儲存臉部 ROI
    locations = []   # 儲存臉部框位置
    predictions = [] # 儲存預測結果 (mask, without_mask)

    # 迭代偵測結果
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 過濾掉低信賴度的偵測
        if confidence > confidence_threshold:
            # 計算臉部邊界框座標 (相對於原始圖片)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 確保邊界框在圖片範圍內
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # 提取臉部 ROI
            face = image[startY:endY, startX:endX]
            # Keras 模型通常需要 BGR 轉 RGB
            if face.size == 0: continue # 避免空的 ROI
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # 調整大小以符合模型輸入
            face_resized = cv2.resize(face_rgb, (224, 224))

            face_preprocessed = preprocessing.image.img_to_array(face_resized)
            face_preprocessed = applications.mobilenet_v2.preprocess_input(face_preprocessed)
            face_preprocessed = np.expand_dims(face_preprocessed, axis=0) # 增加 batch 維度

            faces.append(face_preprocessed)
            locations.append((startX, startY, endX, endY))

    # --- 口罩分類預測 (如果偵測到臉部) ---
    if len(faces) > 0:
        # 一次性進行所有臉部的預測，效率更高
        predictions = mask_model.predict(np.vstack(faces), batch_size=32)

    # --- 在圖片上繪製結果 ---
    for (box, pred) in zip(locations, predictions):
        (startX, startY, endX, endY) = box
        # 取得預測結果
        (mask_prob, without_mask_prob) = pred

        # 決定標籤和顏色
        label = "Mask" if mask_prob > without_mask_prob else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # BGR

        # 顯示機率
        label_text = f"{label}: {max(mask_prob, without_mask_prob) * 100:.2f}%"

        # 繪製邊界框和標籤
        cv2.putText(image, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    # 插入的 Matplotlib 顯示程式碼
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig_for_display, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detection Result')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    output_filename = "Q4_mask_detected.png"
    cv2.imwrite(output_filename, image)

# --- 主程式 ---
if __name__ == "__main__":
    image_file = "../img/image3.png"
    detect_face_mask(image_file)