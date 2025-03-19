import cv2
import numpy as np

# 讀取影像
img = cv2.imread("img/antar.jpg")

# 使用 3x3 中值濾波器進行降噪
median_filtered = cv2.medianBlur(img, 3)

# 使用 3x3 高斯濾波器進行降噪
gaussian_filtered = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

# 分別顯示原始影像、中值濾波器與高斯濾波器處理結果
cv2.imshow("Original Image", img)
cv2.imshow("Median Filter", median_filtered)
cv2.imshow("Gaussian Filter", gaussian_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
