import cv2
import numpy as np

# --- 影像檔案名稱 ---
image_file = "img/myhand.jpg"

# --- 讀取影像 ---
img = cv2.imread(image_file)

# --- 影像前處理 ---
# 1. 轉換為灰度圖
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 應用二值化閾值處理
#    cv2.THRESH_BINARY 表示高於閾值的像素設為 maxval (255)，低於的設為 0
ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# --- 尋找輪廓 ---
# cv2.RETR_EXTERNAL: 只檢測最外層的輪廓
# cv2.CHAIN_APPROX_SIMPLE: 壓縮水平、垂直和對角線段，只留下它們的端點
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# --- 繪製輪廓 ---
# 建立一個原始影像的副本，以便在其上繪製輪廓而不修改原始影像
contour_img = img.copy()
# 使用綠色 (0, 255, 0) 和厚度 2 來繪製所有找到的輪廓 (-1 表示繪製所有輪廓)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# --- 顯示結果 ---
cv2.imshow('Original Image (myhand.jpg)', img)
cv2.imshow('Detected Contours', contour_img)

cv2.imwrite('Result image/Ch15_Q6_contours.jpg', contour_img) # 儲存結果影像

cv2.waitKey(0)
cv2.destroyAllWindows()