import cv2
import numpy as np

# --- 讀取影像 ---
image_file = "img/temple.jpg"
src = cv2.imread(image_file)

# --- 建立形態學核心 (Kernel) ---
# 使用 getStructuringElement() 函數自定義核心
# cv2.MORPH_RECT 表示建立一個矩形核心
# (3,3) 表示核心的大小為 3x3
# 其他選項還有 cv2.MORPH_ELLIPSE (橢圓), cv2.MORPH_CROSS (十字形)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


# --- 應用形態學梯度 ---
# 形態學梯度 = 影像膨脹(dilation)結果 - 影像腐蝕(erosion)結果
dst = cv2.morphologyEx(src, cv2.MORPH_GRADIENT, kernel)

# --- 顯示結果 ---
cv2.imshow("Original Temple Image (src)", src)
cv2.imshow("Morphological Gradient (Edges)", dst)

cv2.imwrite('Result image/Ch12_Q3_edges.jpg', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()