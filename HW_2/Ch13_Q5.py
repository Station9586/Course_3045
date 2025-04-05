import cv2
import numpy as np

# --- 讀取影像並預處理 ---
image_file = "img/geneva.jpg"
src_orig = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE) # 黑白讀取

# 應用高斯模糊降低噪音 (Laplacian 對噪音很敏感)
src = cv2.GaussianBlur(src_orig, (3,3), 0)

# ksize = 1 (使用特定的 3x3 核心)
# 核心: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
dst_tmp_k1 = cv2.Laplacian(src, cv2.CV_32F, ksize=1)
dst_lap_k1 = cv2.convertScaleAbs(dst_tmp_k1) # 將負值轉正值

# ksize = 3 (原始範例使用的大小)
dst_tmp_k3 = cv2.Laplacian(src, cv2.CV_32F, ksize=3)
dst_lap_k3 = cv2.convertScaleAbs(dst_tmp_k3) # 將負值轉正值

# ksize = 5
dst_tmp_k5 = cv2.Laplacian(src, cv2.CV_32F, ksize=5)
dst_lap_k5 = cv2.convertScaleAbs(dst_tmp_k5) # 將負值轉正值

# --- 輸出影像 ---
cv2.imshow("Src (Blurred)", src)
cv2.imshow("Laplacian ksize=1", dst_lap_k1)
cv2.imshow("Laplacian ksize=3", dst_lap_k3)
cv2.imshow("Laplacian ksize=5", dst_lap_k5)

cv2.imwrite('Result image/Ch13_Q5_laplacian_ksize1.jpg', dst_lap_k1)
cv2.imwrite('Result image/Ch13_Q5_laplacian_ksize3.jpg', dst_lap_k3)
cv2.imwrite('Result image/Ch13_Q5_laplacian_ksize5.jpg', dst_lap_k5)


cv2.waitKey(0)
cv2.destroyAllWindows()