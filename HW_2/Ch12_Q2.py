import cv2
import numpy as np

src = cv2.imread("img/j.jpg")
kernel = np.ones((3,3),np.uint8)                        # 建立3x3內核
dst = cv2.morphologyEx(src,cv2.MORPH_GRADIENT,kernel)   # gradient

cv2.imshow("src",src)
cv2.imshow("after morpological gradient",dst)

cv2.imwrite('Result image/Ch12_Q2_j.jpg', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
