import cv2
import numpy as np
import imageio

# load image
image = cv2.imread('img/image.png')
# 將圖片縮小為原來尺寸的 0.5 倍
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# 取得圖片的高度和寬度
height, width = image.shape[:2]
# 建立一個與圖片尺寸相同的黑色遮罩 (mask)，資料型態為無符號 8 位元整數 (灰度圖像)
mask = np.zeros((height, width), dtype=np.uint8)

# 設定聚光燈的半徑，取圖片較小邊長度的 1/4
radius = min(height, width) >> 2

# 設定聚光燈的起始 x 座標，使其從圖片右側外開始移動
start_x = width + radius
# 設定聚光燈的中心 y 座標，位於圖片垂直方向的中央
center_y = height >> 1

# 建立一個名為 'Moving Spotlight' 的視窗，並允許調整視窗大小
cv2.namedWindow('Moving Spotlight', cv2.WINDOW_NORMAL)

# 建立一個空的列表，用於儲存動畫的每一幀
frames = []
# 設定動畫的總幀數
N = 190

# 開始動畫的迴圈，執行 N 次
for _ in range(N):
    # 在每一幀開始時，將遮罩填滿黑色 (數值 0)，以清除上一幀的聚光燈
    mask.fill(0)
    # 計算目前聚光燈的中心 x 座標
    # start_x 會遞減，透過模除運算使其在圖片寬度加上兩倍半徑的範圍內循環
    # 然後減去半徑，使得聚光燈從完全在右側外開始移動，到完全在左側外結束
    center_x = start_x % (width + radius << 1) - radius 

    # 在遮罩上畫一個白色的圓形，代表聚光燈
    # 圓心為 (center_x, center_y)，半徑為 radius，顏色為白色 (255)，-1 表示填充圓形
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # 使用位元 AND 運算將遮罩應用到原始圖片上
    # 只有在遮罩中為白色 (非零) 的區域，原始圖片的像素才會保留下來，形成聚光燈效果
    result = cv2.bitwise_and(image, image, mask=mask)
    # 在名為 'Moving Spotlight' 的視窗中顯示處理後的結果圖片
    cv2.imshow('Moving Spotlight', result)

    # 將目前幀的顏色空間從 BGR (OpenCV 預設) 轉換為 RGB (ImageIO 需要的格式)
    fram_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # 將轉換後的 RGB 幀添加到 frames 列表中
    frames.append(fram_rgb)
    # 將聚光燈的起始 x 座標向左移動 10 個像素，使其在下一幀中向左移動
    start_x -= 10

    # 如果聚光燈已經完全移到圖片左側外，則將其重置到圖片右側外，實現循環移動的效果
    if start_x < -radius:
        start_x = width + radius

    # 等待 30 毫秒，並檢查是否有按下按鍵
    # 如果按下的是 ESC 鍵 (ASCII 碼為 27)，則跳出迴圈，結束動畫
    if cv2.waitKey(30) == 27:
        break

# 關閉所有 OpenCV 建立的視窗
cv2.destroyAllWindows()

# 使用 ImageIO 將 frames 列表中的所有幀儲存為一個 GIF 動畫檔案 'ch08_19.gif'
# duration 參數設定每一幀的顯示時間為 0.033 秒 (約 30 FPS)
imageio.mimsave('ch08_19.gif', frames, duration=0.033)