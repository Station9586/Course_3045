import cv2
import numpy as np

# --- 檔案名稱 ---
template_file = "img/template.jpg"
target_file = "img/hw15_2.jpg"

# --- 載入影像 ---
template_img_color = cv2.imread(template_file)
target_img_color = cv2.imread(target_file)


# --- 預處理：取得模板輪廓 ---
template_gray = cv2.cvtColor(template_img_color, cv2.COLOR_BGR2GRAY)
# 二值化：將白色星星變為前景，黑色背景變為背景
ret_t, thresh_t = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)
# 尋找輪廓
contours_t, hierarchy_t = cv2.findContours(thresh_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 假設模板只有一個主要輪廓（星星），選擇最大的那個
if contours_t:
    template_contour = max(contours_t, key=cv2.contourArea)
else:
    print("錯誤：在模板影像中找不到輪廓。")
    exit()

# --- 預處理：取得目標影像輪廓 ---
target_gray = cv2.cvtColor(target_img_color, cv2.COLOR_BGR2GRAY)
# 二值化
ret_h, thresh_h = cv2.threshold(target_gray, 127, 255, cv2.THRESH_BINARY)
# 尋找輪廓
contours_h, hierarchy_h = cv2.findContours(thresh_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- 比對輪廓 ---
min_diff = float('inf') # 初始化最小差異為無限大
best_match_contour = None

for contour in contours_h:
    # 計算當前輪廓與模板輪廓的形狀相似度
    diff = cv2.matchShapes(template_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)

    print(f" - 輪廓差異度: {diff:.4f}")

    # 如果當前差異度比已記錄的最小差異度還小，則更新最小差異度和最佳匹配輪廓
    if diff < min_diff:
        min_diff = diff
        best_match_contour = contour

# --- 在目標影像上標記結果 ---
result_img = target_img_color.copy() # 複製一份原始彩色影像來繪製結果

if best_match_contour is not None:
    print(f"\n找到最相似的輪廓，差異度為: {min_diff:.4f}")
    # 使用綠色 (0, 255, 0) 實心填充最佳匹配的輪廓
    cv2.drawContours(result_img, [best_match_contour], -1, (0, 255, 0), cv2.FILLED)
else:
    print("\n錯誤：未能找到任何輪廓進行比對或未能確定最佳匹配。")


# --- 顯示結果 ---
# 顯示帶有綠色填充結果的目標影像
cv2.imshow('Result - Best Match Filled (Green)', result_img)
cv2.imshow('Original Target Image (hw15_2.jpg)', target_img_color)
cv2.imshow('Template Image (template.jpg)', template_img_color)

cv2.imwrite('Result image/Ch15_Q5_best_match.jpg', result_img) # 儲存結果影像

cv2.waitKey(0)
cv2.destroyAllWindows()