import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_edge_detection(image_path):
    # 讀取圖片並轉為灰度圖
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) # Canny 前先去噪

    # --- Sobel Edge Detector ---
    # 分別計算 x 和 y 方向的梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # 計算梯度大小
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    sobel_combined = np.uint8(np.absolute(sobel_combined) / np.max(np.absolute(sobel_combined)) * 255)

    # --- Prewitt Edge Detector ---
    # Prewitt 核心
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # 使用 filter2D 套用核心
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    # 計算梯度大小 (轉為 uint8 顯示)
    prewitt_combined = np.sqrt(prewittx.astype(np.float64)**2 + prewitty.astype(np.float64)**2)
    prewitt_combined = np.uint8(np.absolute(prewitt_combined) / np.max(np.absolute(prewitt_combined)) * 255)


    # --- Laplacian Edge Detector ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = np.uint8(np.absolute(laplacian))

    # --- Canny Edge Detector ---
    # 需要設定兩個閾值
    canny = cv2.Canny(gray, threshold1=100, threshold2=200)

    # --- 使用 Matplotlib 顯示結果 ---
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Detector')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(prewitt_combined, cmap='gray')
    plt.title('Prewitt Edge Detector')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(laplacian_abs, cmap='gray')
    plt.title('Laplacian Edge Detector')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(canny, cmap='gray')
    plt.title('Canny Edge Detector')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 主程式 ---
if __name__ == "__main__":
    image_file = "../img/image.png"
    compare_edge_detection(image_file)