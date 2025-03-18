# 113-2 影像處理(3045)

## 全部的作業內容，單純記錄用

### HW_1
**Ch07_Q5**

設計一個模擬繁星點綴的程式。在黑色背景上隨機生成100顆星星。星星的亮度 (灰階值) 和大小隨機分配。星星實現閃爍效果,讀者可以設定每0.1秒閃爍一次, 模擬夜空中繁星的動態變化。程式執行結果可以參考下方左圖,程式概念與設計 邏輯如下:

**畫布建立**

- 使用 NumPy 建立一個大小為(高度,寬度)的黑色畫布。
- 黑色畫布對應灰階值為0。

**星星屬性**

- 位置:(x,y)座標,隨機生成。
- 半徑:模擬星星的大小,範圍為3~6像素。
- 亮度:模擬星星的明暗,範圍為100~255的灰階值。

**閃爍效果**

- 每幀更新畫布時,隨機調整星星的亮度。
- 亮度在100~255 範圍內隨機變化,讓星星看起來像在閃爍。


**Ch08_19**

參考
```py
import cv2
import numpy as np

# 讀取影像
image = cv2.imread("forest.jpg")
# 創建與影像大小相同的遮罩
mask = np.zeros(image.shape[:2], dtype=np.uint8)    # 單通道遮罩    
# 設定圓的初始位置和移動方向
radius = min(image.shape[0], image.shape[1]) // 4
start_x = image.shape[1] + radius                   # 從右側開始
center_y = image.shape[0] // 2      # 圓的垂直位置固定為影像中央
# 視窗設置
cv2.namedWindow("Moving Mask", cv2.WINDOW_NORMAL)
while True:   
    mask.fill(0)                    # 重置遮罩
    # 計算圓的位置
    center_x = start_x % (image.shape[1] + 2 * radius) - radius
    # 繪製圓形遮罩
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    # 在遮罩中進行處理（僅處理白色區域）
    result = cv2.bitwise_and(image, image, mask=mask)
    # 顯示結果
    cv2.imshow("Moving Mask", result)

    # 更新位置
    start_x -= 10  # 每幀向左移動10個像素
    # 重置到右側重新開始
    if start_x < -radius:
        start_x = image.shape[1] + radius

    # 控制速度與結束條件
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```
用出 (1)聚光燈設計 (2)轉成GIF格式


**Ch9_22**

將原始圖案分解，依照位元觀念，從 $a_0$ 至 $a_7$ 分解為 8 個圖案，在分解過程，將大於 0 的像素值改為 255，最後顯示 8 個影像圖。


**Ch9_25**

將浮水印影像 copyright.jpg，嵌入原始影像，然後再將此影像擷取出來。


**Ch10**

任選一張彩色影像執行（1）縮放（2）放大（3）平移（4）旋轉（5）仿射（6）透視


**Ch11_Q1**

請擴充
```py
import cv2
import numpy as np

src = np.ones((3,3), np.float32) * 150
src[1,1] = 20
print(f"src = \n {src}")
dst = cv2.medianBlur(src, 3)
print(f"dst = \n {dst}")
```
觀察中值濾波器的操作，建立 5x5 的矩陣，從左上到右下對角線值是 20，其他值是 150，請使用 ksize = 3 和 ksize = 5，然後列出所建立的矩陣，以及執行結果。


**Ch11_Q2**

使用中值濾波器和高斯濾波器，和相同大小的 3x3 濾波核，對圖片執行降噪處理，最後列出原始影像、中值濾波器與高斯濾波器處理結果影像。


**Ch11_Q3**

重新設計
```py
import cv2

src = cv2.imread("border.jpg")
dst1 = cv2.blur(src, (3, 3))            # 均值濾波器 - 3x3 濾波核
dst2 = cv2.blur(src, (7, 7))            # 均值濾波器 - 7x7 濾波核

dst3 = cv2.GaussianBlur(src,(3,3),0,0)  # 高斯濾波器 - 3x3 的濾波核
dst4 = cv2.GaussianBlur(src,(7,7),0,0)  # 高斯濾波器 - 7x7 的濾波核

cv2.imshow("dst 3 x 3",dst1)
cv2.imshow("dst 7 x 7",dst2)
cv2.imshow("Gauss dst 3 x 3",dst3)
cv2.imshow("Gauss dst 7 x 7",dst4)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
但是將高斯濾波器改為雙邊濾波器，同時比較 3x3 和 7x7 濾波核，均值濾波器與雙邊濾波器的執行結果。