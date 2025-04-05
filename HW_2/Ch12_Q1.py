import cv2
import numpy as np

# --- 從檔案載入影像 ---
file_path = 'img/snowman.jpg'
image = cv2.imread(file_path)
# --- 影像前處理 ---
# 將影像轉換為灰度圖
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- 邊緣偵測 ---
# 使用 Canny 演算法偵測邊緣
edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)


# 顯示原始影像
cv2.imshow('Original Image', image)

# 顯示邊緣偵測結果
cv2.imshow('Detected Edges (Canny)', edges)

cv2.imwrite('Result image/Ch12_Q1_edges.jpg', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()