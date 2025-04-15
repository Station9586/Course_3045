## 專題計畫：簡易文件掃描器 / 透視校正

### 專案標題 (Project Title):
簡易文件掃描器 / 透視校正 (Simple Document Scanner / Perspective Correction)

專案目標 (Objective):
開發一個 Python 程式，能夠自動偵測輸入圖片中的文件輪廓，並對其進行透視變換，以獲得如同掃描般平整、校正後的文件影像。

核心技術 (Core Technologies):
- Python 程式語言
- OpenCV 函式庫
- NumPy 函式庫
- 影像預處理 (灰階、模糊)
- Canny 邊緣偵測
- 輪廓尋找、排序與近似
- 角點偵測與排序
- 幾何變換 (透視變換)

預計實作步驟 (Implementation Steps):

1.  **影像載入與預處理 (Image Loading and Preprocessing):**
    * 使用 `cv2.imread()` 讀取包含文件的輸入圖片。
    * 使用 `cv2.cvtColor()` 將圖片轉換為灰度圖。
    * 使用 `cv2.GaussianBlur()` 進行高斯模糊以減少雜訊。
    * (可選) 使用 `cv2.resize()` 調整圖片大小以加快後續處理速度。

2.  **邊緣偵測 (Edge Detection):**
    * 使用 `cv2.Canny()` 偵測預處理後影像的邊緣。

3.  **輪廓尋找與篩選 (Contour Finding and Filtering):**
    * 使用 `cv2.findContours()` 在 Canny 邊緣圖上尋找所有輪廓 (建議使用 `cv2.RETR_EXTERNAL` 和 `cv2.CHAIN_APPROX_SIMPLE`)。
    * 使用 `cv2.contourArea()` 計算每個輪廓的面積，並透過 `sorted()` 函數將輪廓按面積從大到小排序。
    * 遍歷排序後的輪廓，尋找第一個近似為四邊形的輪廓：
        * 使用 `cv2.arcLength()` 計算輪廓周長。
        * 使用 `cv2.approxPolyDP()` 進行多邊形近似，判斷是否得到 4 個頂點。

4.  **角點提取與排序 (Corner Extraction and Sorting):**
    * 從步驟 3 找到的近似四邊形輪廓中，提取 4 個角點的座標。
    * 對這 4 個角點進行排序，確保其順序為：左上 (Top-Left), 右上 (Top-Right), 右下 (Bottom-Right), 左下 (Bottom-Left)。 (排序是透視變換成功的關鍵)

5.  **透視變換 (Perspective Transformation):**
    * 定義輸出影像的目標寬度和高度 (可基於原始比例或標準尺寸)。
    * 建立目標影像的 4 個角點座標 (e.g., `[[0, 0], [width, 0], [width, height], [0, height]]`)。
    * 使用原始影像中排序好的 4 個角點和目標影像的 4 個角點，計算透視變換矩陣 `M = cv2.getPerspectiveTransform()`。
    * 使用 `cv2.warpPerspective()` 將變換矩陣 `M` 應用於 **原始彩色圖片**，得到校正後的俯視圖。

6.  **結果展示與儲存 (Display/Save Results):**
    * 使用 `cv2.imshow()` 同時顯示原始圖片和校正後的圖片以供比較。
    * 使用 `cv2.imwrite()` 將校正後的圖片儲存為檔案。

預期輸入 (Expected Input):
一張包含矩形文件（例如 A4 紙、名片、書頁等）的圖片，文件可能存在角度傾斜或透視變形。

預期輸出 (Expected Output):
一張經過校正、內容區域被裁切出來、視角類似於掃描器的文件圖片。

潛在挑戰與可擴充功能 (Potential Challenges / Extensions):
* 挑戰：在複雜背景中準確找到文件輪廓；準確排序 4 個角點。
* 擴充：對輸出的文件影像自動進行二值化或亮度/對比度增強；處理多個文件的情況。
