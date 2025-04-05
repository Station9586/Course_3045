import cv2
import numpy as np

# --- 影像檔案名稱 ---
image_file = "img/macau.jpg"

# --- 讀取影像 ---
src = cv2.imread(image_file)

# --- 影像前處理 ---
# 轉換為灰度圖，Canny 需要單通道影像
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# --- 設定 Canny 參數 ---
minVal = 50
maxVal = 100

# --- Canny 邊緣偵測 ---

# 1. 使用 L2gradient = False (預設值)
edges_l1 = cv2.Canny(gray, minVal, maxVal, L2gradient=False)

# 2. 使用 L2gradient = True
edges_l2 = cv2.Canny(gray, minVal, maxVal, L2gradient=True)

# --- 顯示結果 ---
cv2.imshow('Original Macau Image', src)
cv2.imshow('Canny Edges (L1 Norm - L2gradient=False)', edges_l1)
cv2.imshow('Canny Edges (L2 Norm - L2gradient=True)', edges_l2)

cv2.imwrite('Result image/Ch13_Q6_canny_l1.jpg', edges_l1)
cv2.imwrite('Result image/Ch13_Q6_canny_l2.jpg', edges_l2)

cv2.waitKey(0)
cv2.destroyAllWindows()