import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """
    對輪廓的四個角點進行排序：左上、右上、右下、左下。
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    根據給定的四個角點，對影像進行透視變換。
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- 主程式 ---
if __name__ == "__main__":
    image_path = "../img/image7.png"

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"無法讀取圖片： {image_path}")
        orig = image.copy()
    except Exception as e:
        print(f"讀取圖片時發生錯誤: {e}")
        exit()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ========== 調整參數區塊 1 (高斯模糊) ==========
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2. 邊緣偵測 ---
    edged = cv2.Canny(blurred, 50, 150)

    # ========== 調整參數區塊 4 (Closing Kernel Size) ==========
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # --- 3. 尋找文件輪廓 ---
    # 在關閉操作後的影像上尋找輪廓
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # <--- 使用 closed
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # 檢查面積最大的 5 個輪廓
    doc_cnt = None


    for i, c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        epsilon_factor = 0.02
        approx = cv2.approxPolyDP(c, epsilon_factor * peri, True)

        if len(approx) == 4:
            doc_cnt = approx
            break

    # --- 4 & 5. 執行透視變換 ---
    try:
        warped = four_point_transform(orig, doc_cnt.reshape(4, 2))

        # --- 後處理：轉灰度並二值化 ---
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_final = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(warped_final, cmap='gray')
        plt.title('Scanned')
        plt.axis('off')

        plt.suptitle('Scanned Result', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- 儲存結果 ---
        output_path = "scanned_output.jpg"
        cv2.imwrite(output_path, warped_final)
        print(f"成功！校正後的圖片已儲存至 {output_path}")

    except Exception as e:
        print(f"進行透視變換、後處理或顯示時發生錯誤: {e}")
