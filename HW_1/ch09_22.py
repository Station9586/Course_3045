import cv2
import numpy as np

# 讀取灰階影像
img = cv2.imread("img/image.png", cv2.IMREAD_GRAYSCALE)

n, m = img.shape[:2]
title_height = 80  # 標題區域高度

# 用來存放 8 個位元平面影像
bit_planes = []

for k in range(8):
    # 建立一個大小為 (n + title_height) x m 的黑色影像
    tmp = np.zeros((n + title_height, m), dtype=np.uint8)
    # 設定上方標題區為灰色 (數值200)
    tmp[:title_height, :] = 200

    # 取得要顯示位元平面的區域
    plane_region = tmp[title_height:, :]

    # 利用位元運算取出第 k 位元
    # 若該位元為1，則變為 255；否則為 0
    plane = ((img >> k) & 1) * 255
    plane_region[:] = plane

    # 在標題區加入文字，顯示「Bit plane k」
    text = "Bit plane " + str(k + 1)
    # 設定文字位置 (依據影像寬度調整)
    text_pos = (m // 2 - 80, title_height - 20)
    cv2.putText(tmp, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,), 4, cv2.LINE_AA)

    bit_planes.append(tmp)

# 上半部水平拼接前 4 個位元平面，及下半部拼接後 4 個位元平面
top = np.hstack(bit_planes[:4])
bottom = np.hstack(bit_planes[4:])
# 垂直拼接上下兩部分
mosaic = np.vstack((top, bottom))

cv2.imshow("Bitplane", mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()
