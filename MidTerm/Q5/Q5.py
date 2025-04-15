import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_text_watermark(image_path, output_path="Q5_watermarked_output.jpg",
                       watermark_text="Watermark",
                       opacity=0.5, position='bottom_right',
                       font_scale=1.0, font_color=(255, 255, 255), thickness=2):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return

    # 創建一個與原圖一樣大小的透明覆蓋層
    overlay = image.copy()
    output = image.copy()
    (h, w) = image.shape[:2]

    # 設定字體
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 取得文字大小
    text_size, _ = cv2.getTextSize(watermark_text, font, font_scale, thickness)
    text_w, text_h = text_size

    # 計算文字位置
    margin = 10
    if position == 'bottom_right':
        x = w - text_w - margin
        y = h - margin
    elif position == 'top_left':
        x = margin
        y = text_h + margin
    elif position == 'top_right':
        x = w - text_w - margin
        y = text_h + margin
    elif position == 'bottom_left':
        x = margin
        y = h - margin
    else: # 預設右下角
        x = w - text_w - margin
        y = h - margin

    # 在覆蓋層上繪製文字 (不透明)
    cv2.putText(overlay, watermark_text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # 使用 addWeighted 將覆蓋層（帶有文字）與原始圖片混合
    # alpha 是 overlay 的權重 (即浮水印的不透明度)
    # beta 是原始圖片的權重 (1 - alpha)
    cv2.addWeighted(overlay, opacity, output, 1 - opacity, 0, output)

    # 插入的 Matplotlib 顯示程式碼
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title('Watermarked Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 儲存結果
    cv2.imwrite(output_path, output)


# --- 主程式 ---
if __name__ == "__main__":
    image_file = "../img/image2.png"
    add_text_watermark(image_file,
                       watermark_text="Copy Right",
                       opacity=0.4,
                       font_scale=0.8,
                       font_color=(255, 255, 255)) # 白色半透明浮水印