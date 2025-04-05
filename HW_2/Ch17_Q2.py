import cv2
import numpy as np

# --- 影像檔案名稱 ---
image_file = "img/cloud.jpg"

# --- 讀取影像 ---
img = cv2.imread(image_file)

# --- 影像前處理 ---
# 1. 轉換為灰度圖
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 應用二值化閾值處理
threshold_value = 10
ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# --- 尋找輪廓 ---
# cv2.RETR_EXTERNAL: 只檢測最外層的輪廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 處理主要輪廓 ---
img_result = img.copy() # 創建副本用於繪製

# 假設最大的輪廓是雲
main_contour = max(contours, key=cv2.contourArea)

# --- 1. 計算邊界矩形和寬高比 ---
x, y, w, h = cv2.boundingRect(main_contour)
aspect_ratio = float(w) / h if h != 0 else 0 # 避免除以零
print(f"邊界矩形 (Bounding Box): x={x}, y={y}, width={w}, height={h}")
print(f"寬高比 (Aspect Ratio): {aspect_ratio:.4f}")

# --- 2. 計算輪廓面積、凸包面積和堅實度 ---
contour_area = cv2.contourArea(main_contour)
hull_points = cv2.convexHull(main_contour)
hull_area = cv2.contourArea(hull_points)
solidity = float(contour_area) / hull_area if hull_area != 0 else 0 # 避免除以零
print(f"Solidity: {solidity:.4f}") 

# --- 3. 繪製結果 ---
cv2.drawContours(img_result, [main_contour], -1, (0, 255, 0), 2) # 綠色, 厚度 2
# 用黃色繪製邊界矩形 (BGR: 0, 255, 255)
cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 255), 2) # 厚度 2

# 用紅色繪製凸包 (BGR: 0, 0, 255)
cv2.drawContours(img_result, [hull_points], -1, (0, 0, 255), 2) # 厚度 2

# 顯示結果
cv2.imshow('Original Image (cloud.jpg)', img)
cv2.imshow('Bounding Box (Yellow) & Convex Hull (Red)', img_result)

cv2.imwrite('Result image/Ch17_Q2_bounding_box_hull.jpg', img_result) # 儲存結果影像

cv2.waitKey(0)
cv2.destroyAllWindows()