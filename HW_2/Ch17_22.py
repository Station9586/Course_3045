import cv2
import numpy as np

# 建立一個空白畫布
canvas_size = (200, 200)
canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

# 定義初始輪廓, 圓形
center = (canvas_size[1] // 2, canvas_size[0] // 2)
radius = 50
color = (255, 255, 255)             # 白色
thickness = 2

# 建立動畫
num_frames = 100
for i in range(num_frames):
    # 清空畫布
    canvas.fill(0)
    
    # 動態調整半徑, 實現膨脹與收縮效果
    current_radius = radius + int(30 * np.sin(2 * np.pi * i / num_frames))
    
    # 繪製動態輪廓
    cv2.circle(canvas, center, current_radius, color, thickness)
    
    # 顯示動畫畫面
    cv2.imshow('Contour Animation', canvas)
    if cv2.waitKey(30) & 0xFF == 27:            # 按下 ESC 退出
        break

cv2.destroyAllWindows()