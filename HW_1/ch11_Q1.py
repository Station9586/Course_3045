import cv2
import numpy as np

# 建立 5x5 的矩陣，預設所有值為 150
src = np.ones((5,5), np.float32) * 150

# 將從左上到右下的對角線值設為 20
np.fill_diagonal(src, 20)

print("src:")
print(src)

# 使用 ksize = 3 的中值濾波
dst3 = cv2.medianBlur(src, 3)
print("\n中值濾波 (ksize = 3), dst:")
print(dst3)

# 使用 ksize = 5 的中值濾波
dst5 = cv2.medianBlur(src, 5)
print("\n中值濾波 (ksize = 5), dst:")
print(dst5)
