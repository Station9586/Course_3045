import cv2

src = cv2.imread("img/geneva.jpg",cv2.IMREAD_GRAYSCALE)   # 黑白讀取
src = cv2.GaussianBlur(src,(3,3),0)             # 降低噪音
# Sobel()函數
dstx = cv2.Sobel(src, cv2.CV_32F, 1, 0)         # 計算 x 軸影像梯度
dsty = cv2.Sobel(src, cv2.CV_32F, 0, 1)         # 計算 y 軸影像梯度
dstx = cv2.convertScaleAbs(dstx)                # 將負值轉正值
dsty = cv2.convertScaleAbs(dsty)                # 將負值轉正值
dst_sobel =  cv2.addWeighted(dstx, 0.5,dsty, 0.5, 0)    # 影像融合
# Scharr()函數
dstx = cv2.Scharr(src, cv2.CV_32F, 1, 0)        # 計算 x 軸影像梯度
dsty = cv2.Scharr(src, cv2.CV_32F, 0, 1)        # 計算 y 軸影像梯度
dstx = cv2.convertScaleAbs(dstx)                # 將負值轉正值
dsty = cv2.convertScaleAbs(dsty)                # 將負值轉正值
dst_scharr =  cv2.addWeighted(dstx, 0.5,dsty, 0.5, 0)   # 影像融合
# Laplacian()函數
dst_tmp = cv2.Laplacian(src, cv2.CV_32F,ksize=3)    # Laplacian邊緣影像
dst_lap = cv2.convertScaleAbs(dst_tmp)          # 將負值轉正值
# Canny()函數
dst_canny = cv2.Canny(src, 50, 100)             # minVal=50, maxVal=100
# 輸出影像梯度
cv2.imshow("Canny", dst_canny)
cv2.imshow("Sobel", dst_sobel)
cv2.imshow("Scharr", dst_scharr)
cv2.imshow("Laplacian", dst_lap)

cv2.imwrite('Result image/Ch13_13_sobel.jpg', dst_sobel)
cv2.imwrite('Result image/Ch13_13_scharr.jpg', dst_scharr)
cv2.imwrite('Result image/Ch13_13_laplacian.jpg', dst_lap)
cv2.imwrite('Result image/Ch13_13_canny.jpg', dst_canny)

cv2.waitKey(0)
cv2.destroyAllWindows()
