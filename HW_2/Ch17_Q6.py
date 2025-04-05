import cv2
import numpy as np

# --- 讀取圖片 ---
image_file = "img/cloud.jpg"
image = cv2.imread(image_file)

# 獲取原始影像尺寸
orig_height, orig_width = image.shape[:2]

# 動畫參數
frame_counter = 0
cycle_length = 120

# 無限迴圈動畫
while True:
    # --- 計算當前幀的縮放比例 ---
    # 使用正弦函數產生在 0.5 到 1.0 之間平滑變化的比例
    phase = (2 * np.pi * frame_counter) / cycle_length
    scale_factor = 0.5 + 0.5 * ((np.sin(phase - np.pi / 2) + 1) / 2)

    # 計算當前幀的目標寬度和高度
    current_width = int(orig_width * scale_factor)
    current_height = int(orig_height * scale_factor)

    # 確保尺寸至少為 1x1
    current_width = max(1, current_width)
    current_height = max(1, current_height)

    # --- 縮放影像 ---
    resized_image = cv2.resize(image, (current_width, current_height), interpolation=cv2.INTER_LINEAR)

    # --- 建立顯示畫布 ---
    canvas = np.zeros_like(image) # 建立與原始影像等大的黑色畫布

    # --- 計算置中貼上的座標 ---
    # 計算左上角貼上的起始 x, y 座標，使其置中
    start_x = (orig_width - current_width) // 2
    start_y = (orig_height - current_height) // 2

    # 計算右下角貼上的結束 x, y 座標
    end_x = start_x + current_width
    end_y = start_y + current_height

    # 使用陣列切片將 resized_image 複製到 canvas 的計算位置
    # 確保索引在畫布範圍內
    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_y = min(orig_height, end_y)
    end_x = min(orig_width, end_x)

    # 同時也要確保 resized_image 的對應區域不超過其邊界
    paste_h = end_y - start_y
    paste_w = end_x - start_x

    canvas[start_y:end_y, start_x:end_x] = resized_image[0:paste_h, 0:paste_w]


    # --- 顯示動畫畫面 ---
    cv2.imshow('Centered Resizing Animation (Press ESC to exit)', canvas)

    # --- 等待與檢查按鍵 ---
    key = cv2.waitKey(30) & 0xFF # 等待 30 毫秒
    if key == 27: # ESC 鍵
        break

    # 更新計數器
    frame_counter += 1

# 關閉所有視窗
cv2.destroyAllWindows()