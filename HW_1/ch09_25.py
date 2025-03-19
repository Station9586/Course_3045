import cv2
import numpy as np

# 讀取原始影像與水印影像 (灰階)
orig = cv2.imread("img/image.png", cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread("img/copyright.png", cv2.IMREAD_GRAYSCALE)


h, w = orig.shape

# 調整水印尺寸（嵌入在右下角），此處設定為原圖寬度的 1/4
wm_h, wm_w = watermark.shape
if wm_w > w or wm_h > h:
    new_w = w >> 2
    new_h = int(wm_h * new_w / wm_w)
    watermark = cv2.resize(watermark, (new_w, new_h))
    wm_h, wm_w = watermark.shape

# 將水印轉為 0 ~ 3 的級別 (量化至 4 級)
watermark_quant = (watermark >> 6).astype(np.uint8)  # 256/4 = 64

# 複製原始影像作為嵌入後影像
watermarked = orig.copy()

# 嵌入區域：右下角 ROI
roi_x = w - wm_w
roi_y = h - wm_h

roi = watermarked[roi_y: roi_y + wm_h, roi_x: roi_x + wm_w]
# 清除 ROI 中的兩個最低有效位元 (0xFC = 11111100)
roi_cleared = roi & 0xFC

# 將 watermark_quant 嵌入
roi_watermarked = roi_cleared | watermark_quant

watermarked[roi_y: roi_y + wm_h, roi_x: roi_x + wm_w] = roi_watermarked

# 提取水印：從水印嵌入後影像中取出 ROI，再擷取最低兩位
extracted_quant = watermarked[roi_y: roi_y + wm_h, roi_x: roi_x + wm_w] & 0x03
# 將 0~3 的值映射到 0~255 (乘上約85)
extracted = extracted_quant * 85

# 顯示影像
cv2.imshow("Original", orig)
cv2.imshow("Watermark", watermark)
cv2.imshow("Watermarked", watermarked)
cv2.imshow("Extracted Watermark", extracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
