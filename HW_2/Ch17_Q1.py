import cv2
import numpy as np

# --- 影像檔案名稱 ---
image_file = "img/hand.jpg"

# --- 讀取影像 ---
img = cv2.imread(image_file)

# --- 影像前處理 ---
# 1. 轉換為灰度圖
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 應用二值化閾值處理
threshold_value = 20 
ret, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)


# --- 尋找最外層輪廓 ---
# cv2.RETR_EXTERNAL: 只檢測最外層的輪廓
# cv2.CHAIN_APPROX_SIMPLE: 壓縮輪廓點
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 尋找並標示極值點 ---
img_result = img.copy() # 創建副本用於繪製

main_contour = max(contours, key=cv2.contourArea)

# 尋找極值點
# argmin() / argmax() 會返回最小/最大值的索引
# [:, :, 0] 選取所有點的 x 座標
# [:, :, 1] 選取所有點的 y 座標
leftmost = tuple(main_contour[main_contour[:, :, 0].argmin()][0])
rightmost = tuple(main_contour[main_contour[:, :, 0].argmax()][0])
topmost = tuple(main_contour[main_contour[:, :, 1].argmin()][0])
bottommost = tuple(main_contour[main_contour[:, :, 1].argmax()][0])

# 列出座標
print(f"最左點 (Leftmost):   {leftmost}")
print(f"最右點 (Rightmost):  {rightmost}")
print(f"最上點 (Topmost):    {topmost}")
print(f"最下點 (Bottommost): {bottommost}")

# 在影像上繪製極值點
circle_radius = 7
# 最上與最下點用黃色 (BGR: 0, 255, 255)
cv2.circle(img_result, topmost, circle_radius, (0, 255, 255), -1)
cv2.circle(img_result, bottommost, circle_radius, (0, 255, 255), -1)
# 最左與最右點用藍色 (BGR: 0, 255, 0)
cv2.circle(img_result, leftmost, circle_radius, (0, 255, 0), -1)
cv2.circle(img_result, rightmost, circle_radius, (0, 255, 0), -1)

# 顯示結果
cv2.imshow('Original Image (hand.jpg)', img)
cv2.imshow('Extreme Points Marked', img_result)

cv2.imwrite('Result image/Ch17_Q1_extreme_points.jpg', img_result) # 儲存結果影像

cv2.waitKey(0)
cv2.destroyAllWindows()