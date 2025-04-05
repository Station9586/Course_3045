# ch14_9.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 載入影像
image_path = 'img/hung.jpg'                        
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將影像轉換為 RGB 格式

# 建立高斯金字塔, 用於生成不同解析度的影像層次, 第一層為原始影像
gaussian_pyramid = [image]
for i in range(3):                              # 下採樣三次, 生成三層
    image = cv2.pyrDown(image)                  # 每次將影像尺寸縮小一半
    gaussian_pyramid.append(image)              # 儲存高斯金字塔

# 建立拉普拉斯金字塔, 用於提取影像的高頻細節
laplacian_pyramid = []
for i in range(2, -1, -1):                      # 從小層到大層生成
    size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
    # 向上採樣恢復尺寸
    expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
    # 轉換為浮點數, 避免數據溢出
    expanded = expanded.astype(np.float32)
    # 計算高頻細節
    laplacian = cv2.subtract(gaussian_pyramid[i].astype(np.float32), expanded)  
    laplacian_pyramid.append(laplacian)

# 強化拉普拉斯層次, 放大高頻細節, 增強影像細節效果
laplacian_pyramid_enhanced = [lap * 3.0 for lap in laplacian_pyramid]

# 重建影像, 從最小層開始逐層重建影像
reconstructed_image = gaussian_pyramid[-1].astype(np.float32)  
for laplacian in laplacian_pyramid_enhanced:
    size = (laplacian.shape[1], laplacian.shape[0])
    # 向上採樣
    reconstructed_image = cv2.pyrUp(reconstructed_image,
                                    dstsize=size).astype(np.float32)
    # 疊加高頻細節
    reconstructed_image = cv2.add(reconstructed_image, laplacian) 

# 將重建影像轉換回 uint8 格式以便顯示
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# 將重建影像與原始影像混合, 以恢復更多細節
alpha = 0.6                                     # 原始影像的權重
beta = 0.4                                      # 重建影像的權重
final_image = cv2.addWeighted(gaussian_pyramid[0], alpha,
                              reconstructed_image, beta, 0)

# 顯示結果
plt.figure(figsize=(10, 10))

# 原始影像
plt.subplot(1, 3, 1)
plt.imshow(gaussian_pyramid[0])
plt.title("Original")
plt.axis("off")

# 向下採樣後的影像
plt.subplot(1, 3, 2)
plt.imshow(gaussian_pyramid[-1])
plt.title("Downsampled")
plt.axis("off")

# 重建並增強細節的影像
plt.subplot(1, 3, 3)
plt.imshow(final_image)
plt.title("Reconstructed and Enhanced")
plt.axis("off")

plt.tight_layout()
plt.show()

# 儲存結果影像
cv2.imwrite('Result image/Ch14_Q3_enhanced.jpg', final_image)
cv2.imwrite('Result image/Ch14_Q3_downsampled.jpg', gaussian_pyramid[-1])