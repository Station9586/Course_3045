import cv2

# 讀取原始影像
src = cv2.imread("img/border.jpg")

# 均值濾波
mean_3 = cv2.blur(src, (3, 3))
mean_7 = cv2.blur(src, (7, 7))

# 雙邊濾波
bilat_3 = cv2.bilateralFilter(src, 3, sigmaColor=75, sigmaSpace=75)
bilat_7 = cv2.bilateralFilter(src, 7, sigmaColor=150, sigmaSpace=150)

# 顯示結果
cv2.imshow("Original", src)
cv2.imshow("Mean Filter 3x3", mean_3)
cv2.imshow("Mean Filter 7x7", mean_7)
cv2.imshow("Bilateral Filter 3x3", bilat_3)
cv2.imshow("Bilateral Filter 7x7", bilat_7)

cv2.waitKey(0)
cv2.destroyAllWindows()