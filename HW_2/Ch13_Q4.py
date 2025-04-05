import cv2
import numpy as np

# --- 讀取影像 ---
image_file = "img/eagle.jpg"
src = cv2.imread(image_file)

# --- 影像前處理 ---
# 將影像轉換為灰度圖
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# --- 使用 Sobel 算子偵測邊緣 ---
# 計算 X 方向梯度
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
abs_sobel_x = cv2.convertScaleAbs(sobel_x) # 取絕對值並轉回 uint8

# 計算 Y 方向梯度
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
abs_sobel_y = cv2.convertScaleAbs(sobel_y) # 取絕對值並轉回 uint8

# 合併 X 和 Y 方向的梯度 (權重各半)
sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# --- 使用 Scharr 算子偵測邊緣 ---
# Scharr 算子是 Sobel 的改良版 (固定 3x3 核心)
# 計算 X 方向梯度
scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
abs_scharr_x = cv2.convertScaleAbs(scharr_x)

# 計算 Y 方向梯度
scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
abs_scharr_y = cv2.convertScaleAbs(scharr_y)

# 合併 X 和 Y 方向的梯度
scharr_combined = cv2.addWeighted(abs_scharr_x, 0.5, abs_scharr_y, 0.5, 0)

# --- 使用 cv2.imshow 顯示結果 ---
cv2.imshow('Original Eagle Image', src)
cv2.imshow('Sobel Edges (ksize=3)', sobel_combined)
cv2.imshow('Scharr Edges', scharr_combined)

cv2.imwrite('Result image/Ch13_Q4_sobel.jpg', sobel_combined)
cv2.imwrite('Result image/Ch13_Q4_scharr.jpg', scharr_combined)

cv2.waitKey(0)

# --- 關閉所有 OpenCV 視窗 ---
cv2.destroyAllWindows()