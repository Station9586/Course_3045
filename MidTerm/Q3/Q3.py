import cv2
import numpy as np
import matplotlib.pyplot as plt

def average_hash(image, hash_size=8):
    # 縮放圖片
    resized = cv2.resize(image, (hash_size, hash_size))
    # 計算像素平均值
    mean_val = np.mean(resized)
    # 計算雜湊：像素值 > 平均值 為 1，否則為 0
    ahash = (resized > mean_val).astype(np.uint8)
    return ahash

def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)

def compare_image_similarity(image_path1, image_path2, hash_size=8, similarity_threshold=10):
    # 讀取圖片並轉為灰度
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("錯誤：無法讀取其中一張或兩張圖片。")
        return

    # 計算雜湊值
    hash1 = average_hash(img1, hash_size)
    hash2 = average_hash(img2, hash_size)

    # 計算漢明距離
    distance = hamming_distance(hash1, hash2)

    print(f"平均雜湊大小: {hash_size}x{hash_size}")
    print(f"漢明距離: {distance}")

    # 判斷相似度
    if distance <= similarity_threshold:
        print(f"結論：圖片相似 (漢明距離 <= {similarity_threshold})")
    else:
        print(f"結論：圖片不相似 (漢明距離 > {similarity_threshold})")


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.imread(image_path1))
    plt.title('Image 1')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.imread(image_path2))
    plt.title('Image 2')
    plt.axis('off')
    plt.suptitle(f'Hamming Distance: {distance}')
    plt.show()


# --- 主程式 ---
if __name__ == "__main__":
    image_file1 = "../img/image.png"  # 替換成第一張圖片路徑
    image_file2 = "../img/image8.png"  # 替換成第二張圖片路徑
    compare_image_similarity(image_file1, image_file2, similarity_threshold=5) # 閾值可調整