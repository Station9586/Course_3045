import cv2

src = cv2.imread("img/old_building.jpg")    # 讀取影像
G0 = src
G1 = cv2.pyrDown(G0)                    # 第 1 次向下採樣
G2 = cv2.pyrDown(G1)                    # 第 2 次向下採樣
G3 = cv2.pyrDown(G2)                    # 第 3 次向下採樣
L0 = G0 - cv2.pyrUp(G1)                 # 建立第 0 層拉普拉斯金字塔
L1 = G1 - cv2.pyrUp(G2)                 # 建立第 1 層拉普拉斯金字塔
L2 = G2 - cv2.pyrUp(G3)                 # 建立第 2 層拉普拉斯金字塔
print(f"L0.shape = \n{L0.shape}")       # 列印第 0 層拉普拉斯金字塔大小
print(f"L1.shape = \n{L1.shape}")       # 列印第 1 層拉普拉斯金字塔大小
print(f"L2.shape = \n{L2.shape}")       # 列印第 2 層拉普拉斯金字塔大小
cv2.imshow("Laplacian L0",L0)           # 顯示第 0 層拉普拉斯金字塔
cv2.imshow("Laplacian L1",L1)           # 顯示第 1 層拉普拉斯金字塔
cv2.imshow("Laplacian L2",L2)           # 顯示第 2 層拉普拉斯金字塔

cv2.imwrite('Result image/Ch14_Q2_laplacian_L0.jpg', L0) # 儲存第 0 層拉普拉斯金字塔
cv2.imwrite('Result image/Ch14_Q2_laplacian_L1.jpg', L1) # 儲存第 1 層拉普拉斯金字塔
cv2.imwrite('Result image/Ch14_Q2_laplacian_L2.jpg', L2) # 儲存第 2 層拉普拉斯金字塔

cv2.waitKey(0)
cv2.destroyAllWindows()