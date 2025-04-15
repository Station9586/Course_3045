import cv2
import numpy as np
import matplotlib.pyplot as plt # 導入 matplotlib

def cartoonize_image_plt(image_path, output_path="cartoon_output.jpg"):
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return

    # 1. 邊緣遮罩 (使用 adaptiveThreshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5) # 去噪
    # 適應性閾值處理，得到黑白邊緣
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9, C=2)

    # 2. 顏色量化/平滑 (使用雙邊濾波)
    # d: 鄰域直徑, sigmaColor: 顏色空間標準差, sigmaSpace: 座標空間標準差
    color = cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)

    # 3. 將邊緣遮罩應用到顏色平滑後的圖片上
    # 將邊緣轉為 BGR 格式，方便進行 bitwise_and
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_bgr) # 邊緣為白色(255)，非邊緣為黑色(0)。與color做AND會保留非邊緣色彩

    # --- 使用 Matplotlib 顯示結果 ---
    plt.figure(figsize=(10, 8)) # 設定圖形大小

    # Subplot 1: 原始圖片 (OpenCV BGR -> Matplotlib RGB)
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off') # 關閉座標軸

    # Subplot 2: 邊緣遮罩 (灰度圖)
    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray') # 灰度圖需要指定 cmap
    plt.title('Adaptive Threshold')
    plt.axis('off')

    # Subplot 3: 顏色平滑 (OpenCV BGR -> Matplotlib RGB)
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral Filter')
    plt.axis('off')

    # Subplot 4: 卡通化結果 (OpenCV BGR -> Matplotlib RGB)
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    plt.title('Cartoonized Image')
    plt.axis('off')

    plt.tight_layout() # 自動調整子圖參數以適應畫布
    plt.show() # 顯示圖形

    # 儲存結果
    cv2.imwrite(output_path, cartoon)

# --- 主程式 ---
if __name__ == "__main__":
    image_file = "../img/image2.png"
    cartoonize_image_plt(image_file)