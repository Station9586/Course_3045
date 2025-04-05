import cv2
import numpy as np

# --- 讀取圖片 ---
image_file = "img/cloud.jpg"
image = cv2.imread(image_file)

# 獲取原始影像尺寸
orig_height, orig_width = image.shape[:2]
frame_size = (orig_width, orig_height) # 影片幀的大小

# --- 設定影片儲存參數 ---
output_filename = 'Result image/Ch17_Q7_cloud_resizing_animation.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30.0
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

# 動畫參數
frame_counter = 0
cycle_length = 120

# 無限迴圈動畫
while True:
    # --- 計算當前幀的縮放比例 ---
    phase = (2 * np.pi * frame_counter) / cycle_length
    scale_factor = 0.5 + 0.5 * ((np.sin(phase - np.pi / 2) + 1) / 2)

    current_width = int(orig_width * scale_factor)
    current_height = int(orig_height * scale_factor)
    current_width = max(1, current_width)
    current_height = max(1, current_height)

    # --- 縮放影像 ---
    resized_image = cv2.resize(image, (current_width, current_height), interpolation=cv2.INTER_LINEAR)

    # 1. 對縮放後的圖像進行預處理以找到輪廓
    gray_resized = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # 使用一個較低的閾值，因為圖像已經縮放
    ret, thresh_resized = cv2.threshold(gray_resized, 10, 255, cv2.THRESH_BINARY)
    # 2. 尋找縮放後圖像的輪廓
    contours_resized, _ = cv2.findContours(thresh_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_resized:
        border_thickness = 3
        cv2.drawContours(resized_image, contours_resized, -1, (0, 255, 0), border_thickness)

    # --- 建立顯示畫布 ---
    canvas = np.zeros_like(image) # 建立與原始影像等大的黑色畫布


    start_x = (orig_width - current_width) // 2
    start_y = (orig_height - current_height) // 2
    end_x = start_x + current_width
    end_y = start_y + current_height

    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_y = min(orig_height, end_y)
    end_x = min(orig_width, end_x)

    paste_h = end_y - start_y
    paste_w = end_x - start_x

    # --- 將帶有邊框的縮放影像貼到畫布中央 ---
    canvas[start_y:end_y, start_x:end_x] = resized_image[0:paste_h, 0:paste_w]


    # --- 顯示動畫畫面 ---
    cv2.imshow('Cloud Animation with Border (Press ESC to stop)', canvas)

    # --- 將當前畫布寫入影片檔案 ---
    if out.isOpened():
        out.write(canvas)

    # --- 等待與檢查按鍵 ---
    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key == 27: # ESC 鍵
        break

    # 更新計數器
    frame_counter += 1

# --- 迴圈結束後釋放資源 ---
out.release()
cv2.destroyAllWindows()