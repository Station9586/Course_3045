import cv2
import numpy as np

# 建立一個空白畫布
canvas_size = (200, 200)
canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

# 定義初始輪廓, 圓形
center = (canvas_size[1] // 2, canvas_size[0] // 2)
radius = 50
color = (255, 255, 255)      # 白色
thickness = 2

# 動畫參數
frame_counter = 0
cycle_length = 100

# 使用無限迴圈來持續播放動畫
while True:
    # 清空畫布 (將所有像素設為黑色)
    canvas.fill(0)

    # 動態調整半徑, 實現膨脹與收縮效果
    # 使用 frame_counter 和 cycle_length 來計算正弦波的當前相位
    current_radius = radius + int(30 * np.sin(2 * np.pi * frame_counter / cycle_length))

    # 確保半徑不會變成負數
    current_radius = max(0, current_radius)

    # 繪製動態輪廓 (圓形)
    cv2.circle(canvas, center, current_radius, color, thickness)

    # 顯示動畫畫面
    # 更新視窗標題提示如何退出
    cv2.imshow('Contour Animation (Press ESC to exit)', canvas)

    # 等待 30 毫秒，並檢查是否有按鍵事件
    key = cv2.waitKey(30) & 0xFF

    # 如果按下的鍵是 ESC (ASCII 值為 27)，則跳出迴圈
    if key == 27:
        break

    # 更新計數器以進行下一幀動畫
    frame_counter += 1

cv2.destroyAllWindows()