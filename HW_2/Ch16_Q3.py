import cv2
import numpy as np

# --- 影像檔案名稱 ---
image_file = "img/hand3.jpg"

# --- 讀取影像 ---
img = cv2.imread(image_file)

# --- 影像前處理 ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# --- 尋找輪廓 ---
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 假設最大的輪廓是手
if contours:
    main_contour = max(contours, key=cv2.contourArea)
else:
    print("錯誤：在此影像中找不到輪廓。")
    exit()

# --- 創建副本用於繪製不同結果 ---
dst1 = img.copy() # 用於繪製凸包點
dst2 = img.copy() # 用於繪製凸包線和缺陷點

# --- 計算凸包 ---
# 獲取凸包的點座標 (用於繪製點和線)
hull_points = cv2.convexHull(main_contour, returnPoints=True)

# --- 繪製結果到 dst1 (只畫凸包的點) ---
if hull_points is not None:
    print(f"\n找到 {len(hull_points)} 個凸包點。")
    for point in hull_points:
        # 在 dst1 上用紅色小實心圓標示凸包的每個點
        cv2.circle(dst1, tuple(point[0]), 3, (0, 0, 255), -1) # 半徑3, 紅色, 實心

# --- 繪製結果到 dst2 (畫凸包線 + 缺陷點) ---
# 繪製綠色凸包線條
cv2.drawContours(dst2, [hull_points], -1, (0, 255, 0), 2) # 綠色, 厚度2

# --- 計算並繪製凸缺陷到 dst2 ---
defect_count = 0
if len(main_contour) > 3:
    # 需要凸包點的索引來計算缺陷
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)

    if hull_indices is not None and len(hull_indices) > 3:
        defects = cv2.convexityDefects(main_contour, hull_indices)

        print("\n--- 凸缺陷 (Convexity Defects) ---")
        if defects is not None:
            defect_count = defects.shape[0]
            print(f"總共找到 {defect_count} 個凸缺陷。")
            print("索引:\t 起始點(Idx)\t 結束點(Idx)\t 最遠點(Idx)\t 約略深度(Px)")
            print("-" * 75)
            for i in range(defect_count):
                s, e, f, d = defects[i, 0]
                far = tuple(main_contour[f][0])   # 最遠點
                depth = d / 256.0

                print(f"{i}:\t {s}\t\t {e}\t\t {f}\t\t {depth:.2f}")

                # 在 dst2 的最遠點繪製一個紅色小實心圓
                cv2.circle(dst2, far, 5, (0, 0, 255), -1) # 半徑5, 紅色, 實心
        else:
            print("計算後未發現凸缺陷。")
    else:
        print("凸包點不足或無效，無法計算凸缺陷。")
else:
    print("輪廓點數不足，無法計算凸包或凸缺陷。")


# --- 顯示最終三個結果視窗 ---
cv2.imshow('src', img)
cv2.imshow('dst1 - Hull Points', dst1)
cv2.imshow('dst2 - Hull Lines', dst2)

print(f"\n最終計算出的缺陷數量為: {defect_count}")

cv2.imwrite('Result image/Ch16_Q3_dst1.jpg', dst1) # 儲存凸包點
cv2.imwrite('Result image/Ch16_Q3_dst2.jpg', dst2) # 儲存凸包線和缺陷點

cv2.waitKey(0)
cv2.destroyAllWindows()