import cv2
import numpy as np
import os # 用於儲存圖片時檢查路徑

# --- 影像檔案名稱 ---
image_file = "img/hand3.jpg"

# --- 讀取影像 ---
img = cv2.imread(image_file)

if img is None:
    print(f"錯誤：無法載入影像 '{image_file}'")
    exit()

# --- 影像前處理 ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# --- 尋找輪廓 (包含層級關係) ---
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# --- 尋找外部和內部輪廓 ---
outer_contour = None
inner_contour = None
outer_defect_count = 0  # 初始化外部缺陷數量
inner_defect_count = 0  # 初始化內部缺陷數量

if contours and hierarchy is not None:
    hierarchy = hierarchy[0]
    outer_contour_index = -1
    inner_contour_index = -1

    # 1. 找最外層輪廓
    for i, h in enumerate(hierarchy):
        if h[3] == -1:
            if outer_contour_index == -1 or cv2.contourArea(contours[i]) > cv2.contourArea(contours[outer_contour_index]):
                 outer_contour_index = i

    # 2. 找內部輪廓
    if outer_contour_index != -1:
        outer_contour = contours[outer_contour_index]
        for i, h in enumerate(hierarchy):
            if h[3] == outer_contour_index:
                 inner_contour_index = i
                 inner_contour = contours[inner_contour_index]
                 break
    else:
         print("錯誤：未能找到最外層輪廓。")
else:
    print("錯誤：在此影像中找不到任何輪廓或層級資訊。")
    exit()


dst1_inner = img.copy()
dst2_outer = img.copy()

# --- 處理外部輪廓 (Outer Contour) ---
if outer_contour is not None and len(outer_contour) > 3:
    # 1. 計算外部凸包點和線
    outer_hull_points = cv2.convexHull(outer_contour, returnPoints=True)
    outer_hull_indices = cv2.convexHull(outer_contour, returnPoints=False)

    # 繪製到 dst2 (外部凸包線)
    cv2.drawContours(dst2_outer, [outer_hull_points], -1, (0, 255, 0), 2) # 綠色線

    # 2. 計算並繪製外部凸缺陷到 dst2，並計算數量
    if outer_hull_indices is not None and len(outer_hull_indices) > 3:
        outer_defects = cv2.convexityDefects(outer_contour, outer_hull_indices)
        if outer_defects is not None:
            outer_defect_count = outer_defects.shape[0] # <--- 計算外部缺陷數量
            for i in range(outer_defect_count):
                s, e, f, d = outer_defects[i, 0]
                far = tuple(outer_contour[f][0])
                cv2.circle(dst2_outer, far, 5, (0, 0, 255), -1) # 紅色點 (缺陷點)

# --- 處理內部輪廓 (Inner Contour) ---
if inner_contour is not None and len(inner_contour) > 3:
    # 1. 計算內部凸包點
    inner_hull_points = cv2.convexHull(inner_contour, returnPoints=True)
    inner_hull_indices = cv2.convexHull(inner_contour, returnPoints=False)

    # 繪製到 dst1 (內部凸包線和點)
    if inner_hull_points is not None:
        # 繪製內部凸包線 (綠色)
        cv2.drawContours(dst1_inner, [inner_hull_points], -1, (0, 255, 0), 2)
        # 繪製內部凸包點 (紅色)
        for point in inner_hull_points:
            cv2.circle(dst1_inner, tuple(point[0]), 3, (0, 0, 255), -1)

    # 2. 計算內部凸缺陷數量 (不需要繪製缺陷點，只需要數量)
    if inner_hull_indices is not None and len(inner_hull_indices) > 3:
        inner_defects = cv2.convexityDefects(inner_contour, inner_hull_indices)
        if inner_defects is not None:
            inner_defect_count = inner_defects.shape[0] # <--- 計算內部缺陷數量

# --- 在影像上繪製凸缺陷數量 ---
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
color = (255, 255, 255) # 白色文字
thickness = 1


inner_text = f"Defects: {inner_defect_count}"
print("凸缺陷數量 (內部):", inner_defect_count)
print("凸缺陷數量 (外部):", outer_defect_count)


# --- 顯示最終結果視窗 (使用您指定的 dst1 和 dst2 標題) ---
cv2.imshow('src', img)
cv2.imshow('dst1', dst1_inner) # 顯示內部處理結果，標題為 dst1
cv2.imshow('dst2', dst2_outer) # 顯示外部處理結果，標題為 dst2

cv2.imwrite("Result image/Ch16_Q3_dst1.jpg", dst1_inner)
cv2.imwrite("Result image/Ch16_Q3_dst2.jpg", dst2_outer)

cv2.waitKey(0)
cv2.destroyAllWindows()