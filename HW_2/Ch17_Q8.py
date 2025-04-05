# ch17_24_with_video_output.py
import cv2
import numpy as np

# 讀取影像並轉換為灰階
src = cv2.imread('img/hand.jpg')

cv2.imshow("Source Image", src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 二值化處理
_, binary = cv2.threshold(src_gray, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# 檢查是否找到輪廓
if not contours:
    print("No contours found.")
    cv2.destroyAllWindows() # 如果沒找到輪廓，關閉來源視窗並退出
    exit()

# 提取最大輪廓
cnt = max(contours, key=cv2.contourArea)
mask = np.zeros(src_gray.shape, np.uint8)
mask = cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)

# 在遮罩區域內找最小與最大像素值
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src_gray, mask=mask)
print(f"最小像素值 = {minVal}, 座標 = {minLoc}")
print(f"最大像素值 = {maxVal}, 座標 = {maxLoc}")

# 儲存最大和最小像素點的原始顏色
min_pixel_color = src[minLoc[1], minLoc[0]].tolist()      # (B, G, R)
max_pixel_color = src[maxLoc[1], maxLoc[0]].tolist()      # (B, G, R)

# --- 設定影片儲存參數 ---
output_filename = 'Result image/output_ex17_8.mp4'
frame_height, frame_width = src.shape[:2]
frame_size = (frame_width, frame_height) # 影片幀的大小
fps = 30.0 # 設定影片的幀率 (Frames Per Second)
# 使用 'mp4v' 編碼器來儲存 MP4 檔案
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 建立 VideoWriter 物件
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

# 動畫參數
num_frames_cycle = 100      # 動畫變化的週期長度 (原來的 num_frames)
frame_index = 0
min_radius = 5              # 初始圓半徑
max_radius = 50             # 最大圓半徑
pixel_radius = 2            # 原始像素點的顯示大小


# --- 動畫迴圈 ---
while True:
    # 計算當前半徑
    radius = int(min_radius + (max_radius - min_radius) * \
                 (0.5 + 0.5 * np.sin(2 * np.pi * frame_index / num_frames_cycle)))
    frame_index = (frame_index + 1) % num_frames_cycle # 使用週期長度

    # 創建畫布以顯示動畫 (使用原始影像副本)
    canvas = src.copy()

    # 繪製以最小像素點為中心的放大與縮小圓形 (綠色填充)
    cv2.circle(canvas, minLoc, radius, [0, 255, 0], -1)

    # 繪製以最大像素點為中心的放大與縮小圓形 (紅色填充)
    cv2.circle(canvas, maxLoc, radius, [0, 0, 255], -1)

    # 恢復最小像素點原始顏色和大小 (繪製在脈衝圓之上)
    cv2.circle(canvas, minLoc, pixel_radius, min_pixel_color, -1)
    # 恢復最大像素點原始顏色和大小 (繪製在脈衝圓之上)
    cv2.circle(canvas, maxLoc, pixel_radius, max_pixel_color, -1)

    if out.isOpened():
        out.write(canvas)

    # 顯示動畫
    cv2.imshow('Animation', canvas)

    # 等待按鍵輸入
    key = cv2.waitKey(int(1000/fps)) # 等待時間基於 FPS
    if key == 27:                   # 按下 ESC 鍵退出
        break

out.release() # 釋放 VideoWriter 物件
cv2.destroyAllWindows()