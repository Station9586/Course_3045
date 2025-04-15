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

---

### HW_2
**Ch12_Q1**

使用 `snowman.jpg` ，列出 `snowman.jpg` 的影像邊緣。

**Ch12_Q2**

請建立 $3 \times 3$ 內核，建立 j 影像的邊緣。

**Ch12_Q3**

使用 `getStructuringElement()` 函數自定義的內核，參考 `ch12_17.py` 建立下列 `temple.jpg` 的邊緣影像。
```py
# ch12_17.py
import cv2
import numpy as np

src = cv2.imread("hole.jpg")
kernel = np.ones((3,3),np.uint8)                        # 建立3x3內核
dst = cv2.morphologyEx(src,cv2.MORPH_GRADIENT,kernel)   # gradient

cv2.imshow("src",src)
cv2.imshow("after morpological gradient",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Ch13_11**

瑞士日內瓦建築物的邊緣偵測，這個程式會分別列出原始影像、`Sobel()`、`Scharr()`和 `Laplacian()`函數的執行結果。

**Ch13_13**

重新設計 `Ch13_11.py` ，增加使用 Canny 檢測方法，最後列出 4 種方法的結果並做比較
```py
# ch13_11.py
import cv2

src = cv2.imread("geneva.jpg",cv2.IMREAD_GRAYSCALE)   # 黑白讀取
src = cv2.GaussianBlur(src,(3,3),0)             # 降低噪音
# Sobel()函數
dstx = cv2.Sobel(src, cv2.CV_32F, 1, 0)         # 計算 x 軸影像梯度
dsty = cv2.Sobel(src, cv2.CV_32F, 0, 1)         # 計算 y 軸影像梯度
dstx = cv2.convertScaleAbs(dstx)                # 將負值轉正值
dsty = cv2.convertScaleAbs(dsty)                # 將負值轉正值
dst_sobel =  cv2.addWeighted(dstx, 0.5,dsty, 0.5, 0)    # 影像融合
# Scharr()函數
dstx = cv2.Scharr(src, cv2.CV_32F, 1, 0)        # 計算 x 軸影像梯度
dsty = cv2.Scharr(src, cv2.CV_32F, 0, 1)        # 計算 y 軸影像梯度
dstx = cv2.convertScaleAbs(dstx)                # 將負值轉正值
dsty = cv2.convertScaleAbs(dsty)                # 將負值轉正值
dst_scharr =  cv2.addWeighted(dstx, 0.5,dsty, 0.5, 0)   # 影像融合
# Laplacian()函數
dst_tmp = cv2.Laplacian(src, cv2.CV_32F,ksize=3)    # Laplacian邊緣影像
dst_lap = cv2.convertScaleAbs(dst_tmp)          # 將負值轉正值
# 輸出影像梯度
cv2.imshow("Src", src)
cv2.imshow("Sobel", dst_sobel)
cv2.imshow("Scharr", dst_scharr)
cv2.imshow("Laplacian", dst_lap)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Ch13_Q4**

有一個影像檔案 eagle.jpg，請使用 `Sobel()` 和 `Scharr()` 建立此影像的邊緣，同時列出比較結果，下列由左到右分別是原始影像`eagle.jpg`，使用 `Sobel()` 和 `Scharr()` 函數的執行結果。

**Ch13_Q5**

在程式實例 `ch13_11.py` 的 `Laplacian()` 函數中，我們使用ksize = 3，獲得很好的邊緣影像，請分別使用 ksize = 1,3,5，然後比較結果

**Ch13_Q6**

有一幅澳門酒店的影像，請使用 Canny 邊緣檢測，minVal=50,maxVal=100，請使用 L2gradient 預設 False 和設 L2gradient=True，繪製此酒店的邊緣影像

**Ch14_Q2**

請重新設計前一個程式，只更改讀取的影像檔案 `old_building.jpg`，列出下列拉普拉斯金字塔的結果。

> 前一程式：請擴充設計 ch14_7.py，到第3次向下採樣，同時建立第2層的拉普拉斯金字塔影像

```py
import cv2

src = cv2.imread("pengiun.jpg")         # 讀取影像
G0 = src
G1 = cv2.pyrDown(G0)                    # 第 1 次向下採樣
G2 = cv2.pyrDown(G1)                    # 第 2 次向下採樣

L0 = G0 - cv2.pyrUp(G1)                 # 建立第 0 層拉普拉斯金字塔
L1 = G1 - cv2.pyrUp(G2)                 # 建立第 1 層拉普拉斯金字塔
print(f"L0.shape = \n{L0.shape}")       # 列印第 0 層拉普拉斯金字塔大小
print(f"L1.shape = \n{L1.shape}")       # 列印第 1 層拉普拉斯金字塔大小
cv2.imshow("Laplacian L0",L0)           # 顯示第 0 層拉普拉斯金字塔
cv2.imshow("Laplacian L1",L1)           # 顯示第 1 層拉普拉斯金字塔

cv2.waitKey(0)
cv2.destroyAllWindows()
```


**Ch14_Q3**

請讀者更改 `ch14_9.py`，第31列的放大倍數為3，同時用自己的老照片測試，以體會修復結果。

```py
# ch14_9.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft JhengHei"]
# 載入影像
image_path = 'hung.jpg'                         # 檔名為 hung.jpg
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將影像轉換為 RGB 格式

# 建立高斯金字塔, 用於生成不同解析度的影像層次, 第一層為原始影像
gaussian_pyramid = [image]
for i in range(3):                              # 下採樣三次, 生成三層
    image = cv2.pyrDown(image)                  # 每次將影像尺寸縮小一半
    gaussian_pyramid.append(image)              # 儲存高斯金字塔

# 建立拉普拉斯金字塔, 用於提取影像的高頻細節
laplacian_pyramid = []
for i in range(2, -1, -1):                      # 從小層到大層生成
    size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
    # 向上採樣恢復尺寸
    expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
    # 轉換為浮點數, 避免數據溢出
    expanded = expanded.astype(np.float32)
    # 計算高頻細節
    laplacian = cv2.subtract(gaussian_pyramid[i].astype(np.float32), expanded)  
    laplacian_pyramid.append(laplacian)

# 強化拉普拉斯層次, 放大高頻細節, 增強影像細節效果
laplacian_pyramid_enhanced = [lap * 1.5 for lap in laplacian_pyramid]

# 重建影像, 從最小層開始逐層重建影像
reconstructed_image = gaussian_pyramid[-1].astype(np.float32)  
for laplacian in laplacian_pyramid_enhanced:
    size = (laplacian.shape[1], laplacian.shape[0])
    # 向上採樣
    reconstructed_image = cv2.pyrUp(reconstructed_image,
                                    dstsize=size).astype(np.float32)
    # 疊加高頻細節
    reconstructed_image = cv2.add(reconstructed_image, laplacian) 

# 將重建影像轉換回 uint8 格式以便顯示
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# 將重建影像與原始影像混合, 以恢復更多細節
alpha = 0.6                                     # 原始影像的權重
beta = 0.4                                      # 重建影像的權重
final_image = cv2.addWeighted(gaussian_pyramid[0], alpha,
                              reconstructed_image, beta, 0)

# 顯示結果
plt.figure(figsize=(10, 10))

# 原始影像
plt.subplot(1, 3, 1)
plt.imshow(gaussian_pyramid[0])
plt.title("原始影像")
plt.axis("off")

# 向下採樣後的影像
plt.subplot(1, 3, 2)
plt.imshow(gaussian_pyramid[-1])
plt.title("向下採樣影像")
plt.axis("off")

# 重建並增強細節的影像
plt.subplot(1, 3, 3)
plt.imshow(final_image)
plt.title("重建並增強影像")
plt.axis("off")

plt.tight_layout()
plt.show()
```

**Ch15_Q5**

使用`template.jpg` 影像檔案，然後找出`hw15_2.jpg` 影像檔案中，外形最類似的輪廓，然後將此輪廓用綠色實心填滿，下方中央小圖是 `template.jpg`。

**Ch15_Q6**

有一個 `myhand.jpg` 影像，請建立這個影像的輪廓


**Ch16_Q3**

請使用 `hand3.jpg`，請繪製凸包，同時列出所有的輪廓的缺陷數量和凸缺陷

**Ch17_Q1**

列出 `hand.jpg` 的手形的最左、最右、最上、最下點，同時最上與最下點用黃色，最左與最右點用黑色。註：需使用不同的閾值，同時檢測最外圍輸廓。

**Ch17_Q2**

計算`doud.jpg` 的寬高比和 `Solidity`，同時用紅色繪製凸包，用黃色繪製矩形框。

**Ch17_Q5**

重新設計 `ch17_22.py`，當按下 Esc 鍵後，程式才會結束。
```py
# ch17_22.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
```

**Ch17_Q6**

重新設計ch17_23.py，此程式會讓圖像縮小到原先的一半，再做放大到原先大小。

```py
# ch17_23.py
import cv2
import numpy as np

# 讀取圖片並轉換為灰階
image = cv2.imread("cloud.jpg")  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化處理
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 尋找輪廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# 檢查是否有輪廓
if len(contours) == 0:
    print("未找到輪廓, 請使用包含明顯物件的圖片")
    exit()

# 建立膨脹動畫的核心結構
kernel_size = 5                 # 核的基礎大小
max_dilation = 20               # 最大膨脹次數
min_dilation = 1                # 最小膨脹次數

# 動畫參數
num_frames = 50
frame_index = 0

while True:
    # 計算當前膨脹大小
    dilation_size = min_dilation + int((max_dilation - min_dilation) * \
                    (0.5 + 0.5 * np.sin(2 * np.pi * frame_index / num_frames)))
    frame_index = (frame_index + 1) % num_frames
    
    # 建立動態核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilation_size, dilation_size))
    
    # 對原始二值化影像進行膨脹
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # 在原圖上繪製膨脹後的輪廓
    canvas = image.copy()
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)    
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)  # 用綠色繪製輪廓
    
    # 顯示動畫畫面
    cv2.imshow('Contour Animation', canvas)
    
    # 等待鍵盤輸入
    key = cv2.waitKey(30)       # 每幀等待 30 毫秒
    if key == 27:               # 按下 ESC 退出
        break

cv2.destroyAllWindows()
```

**Ch17_Q7**

請重新設計上一個習題，增加3像素的綠色邊緣，同時用MP4檔案儲存。

**Ch17_Q8**

請擴充 `ch17_24.py`，增加將動態影片輸出為 `output_ex17_8.mp4`。
```py
# ch17_24.py
import cv2
import numpy as np

# 讀取影像並轉換為灰階
src = cv2.imread('hand.jpg')  

cv2.imshow("Source Image", src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 二值化處理
_, binary = cv2.threshold(src_gray, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# 檢查是否找到輪廓
if not contours:
    print("No contours found.")
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
min_pixel_color = src[minLoc[1], minLoc[0]].tolist()        # (B, G, R)
max_pixel_color = src[maxLoc[1], maxLoc[0]].tolist()        # (B, G, R)

# 動畫參數
num_frames = 100                # 動畫總幀數
frame_index = 0
min_radius = 5                  # 初始圓半徑
max_radius = 50                 # 最大圓半徑
pixel_radius = 2                # 原始像素點的顯示大小

while True:
    # 計算當前半徑
    radius = int(min_radius + (max_radius - min_radius) * \
                 (0.5 + 0.5 * np.sin(2 * np.pi * frame_index / num_frames)))
    frame_index = (frame_index + 1) % num_frames

    # 創建畫布以顯示動畫
    canvas = src.copy()

    # 繪製以最小像素點為中心的放大與縮小圓形
    cv2.circle(canvas, minLoc, radius, [0, 255, 0], -1)     # 綠色填充

    # 繪製以最大像素點為中心的放大與縮小圓形
    cv2.circle(canvas, maxLoc, radius, [0, 0, 255], -1)     # 紅色填充

   # 恢復最小像素點原始顏色和大小
    cv2.circle(canvas, minLoc, pixel_radius, min_pixel_color, -1)
    # 恢復最大像素點原始顏色和大小
    cv2.circle(canvas, maxLoc, pixel_radius, max_pixel_color, -1)

    # 顯示動畫
    cv2.imshow('Animation', canvas)

    # 等待按鍵輸入
    key = cv2.waitKey(30)                               # 每幀等待 30 毫秒
    if key == 27:                                       # 按下 ESC 鍵退出
        break

cv2.destroyAllWindows()
```

**Ch17_22**

設計輪廓為圓形的縮放動畫。

---

### MidTerm
1. 請寫程式比較邊緣偵測的方法?( 1. Sobel edge detector  2.Prewitt edge detector 3.Laplacian edge detector 4.Canny edge detector )

2. 請寫程式將影像卡通化
   
3. 請寫程式將影像相似影像偵測器

4. 請寫程式將影像人臉口罩偵測

5. 請寫程式將影像添加可視化浮水印

6. 請自行開發一個影像專題