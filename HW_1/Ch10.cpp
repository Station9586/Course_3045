#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>

using cv::Mat;

int main(void) {
    // load image
    Mat img = cv::imread("img/copyright.png", cv::IMREAD_COLOR);

    // 儲存 Mat 物件 (圖片) 的 vector
    std::vector<Mat> v;
    // 顯示原始圖片，視窗標題為 "Original"
    imshow("Original", img);

    // 宣告將要儲存各種變形後圖片的 Mat 物件
    Mat scaled_down, scaled_up, translated, rotated, AffT, PersT;
    // 定義縮放比例
    double scale_d = 0.8, scale_u = 1.5;
    // 使用線性插值將原始圖片縮小到原來的 0.8 倍
    cv::resize(img, scaled_down, cv::Size(), scale_d, scale_d, cv::INTER_LINEAR);
    // 使用線性插值將原始圖片放大到原來的 1.5 倍
    cv::resize(img, scaled_up, cv::Size(), scale_u, scale_u, cv::INTER_LINEAR);

    // 定義平移的距離 (dx, dy)
    int dx = 50, dy = 100;
    // 建立一個 2x3 的轉換矩陣，用於平移操作
    // [[1, 0, dx],
    //  [0, 1, dy]]
    Mat T = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    // 使用仿射變換將原始圖片平移 (dx, dy) 的距離，輸出圖片大小與原始圖片相同
    cv::warpAffine(img, translated, T, img.size());

    // 定義旋轉的角度和縮放比例
    double angle = 45, scale = 1;
    // 定義旋轉中心點為圖片的中心
    cv::Point2f center(1.0 * img.cols / 2, 1.0 * img.rows / 2);
    // 取得旋轉矩陣，以 center 為中心，旋轉 angle 角度，縮放比例為 scale
    Mat R = cv::getRotationMatrix2D(center, angle, scale);
    // 使用仿射變換將原始圖片旋轉，輸出圖片大小與原始圖片相同
    cv::warpAffine(img, rotated, R, img.size());

    // 定義仿射變換的來源和目標三角形頂點
    cv::Point2f srcTri[3], dstTri[3];
    // 來源三角形的頂點
    srcTri[0] = cv::Point2f(0, 0);
    srcTri[1] = cv::Point2f(img.cols - 1, 0);
    srcTri[2] = cv::Point2f(0, img.rows - 1);

    // 目標三角形的頂點
    dstTri[0] = cv::Point2f(0, img.rows * 0.33f);
    dstTri[1] = cv::Point2f(img.cols * 0.85f, img.rows * 0.25f);
    dstTri[2] = cv::Point2f(img.cols * 0.15f, img.rows * 0.75f);
    // 根據來源和目標三角形取得仿射變換矩陣
    Mat aff_M = cv::getAffineTransform(srcTri, dstTri);
    // 使用仿射變換將原始圖片變形，輸出圖片大小與原始圖片相同
    cv::warpAffine(img, AffT, aff_M, img.size());

    // 定義透視變換的來源和目標四邊形頂點
    cv::Point2f srcQuad[4], dstQuad[4];
    // 來源四邊形的頂點
    srcQuad[0] = cv::Point2f(0, 0);
    srcQuad[1] = cv::Point2f(img.cols - 1, 0);
    srcQuad[2] = cv::Point2f(img.cols - 1, img.rows - 1);
    srcQuad[3] = cv::Point2f(0, img.rows - 1);
    // 目標四邊形的頂點
    dstQuad[0] = cv::Point2f(img.cols * 0.05f, img.rows * 0.2f);
    dstQuad[1] = cv::Point2f(img.cols * 0.95f, img.rows * 0.1f);
    dstQuad[2] = cv::Point2f(img.cols * 0.85f, img.rows * 0.9f);
    dstQuad[3] = cv::Point2f(img.cols * 0.15f, img.rows * 0.85f);
    // 根據來源和目標四邊形取得透視變換矩陣
    Mat pers_M = cv::getPerspectiveTransform(srcQuad, dstQuad);
    // 使用透視變換將原始圖片變形，輸出圖片大小與原始圖片相同
    cv::warpPerspective(img, PersT, pers_M, img.size());

    // 在縮小後的圖片上添加文字 "Scaled Down"
    cv::putText(scaled_down, "Scaled Down", cv::Point(40, 120),
                cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 0), 10);
    // 在放大後的圖片上添加文字 "Scaled Up"
    cv::putText(scaled_up, "Scaled Up", cv::Point(40, 120),
                cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 0), 10);
    // 在平移後的圖片上添加文字 "Translated"
    cv::putText(translated, "Translated", cv::Point(40, 80),
                cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 255, 255), 10);
    // 在旋轉後的圖片上添加文字 "Rotated"
    cv::putText(rotated, "Rotated", cv::Point(40, 120),
                cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(255, 255, 255), 10);
    // 在仿射變換後的圖片上添加文字 "Affine Transformed"
    cv::putText(AffT, "Affine Transformed", cv::Point(40, 120),
                cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(255, 255, 255), 10);
    // 在透視變換後的圖片上添加文字 "Perspective Transformed"
    cv::putText(PersT, "Perspective Transformed", cv::Point(40, 120),
                cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(255, 255, 255), 10);

    // 將所有變形後的圖片依序加入到 vector v 中
    v.emplace_back(scaled_down);
    v.emplace_back(scaled_up);
    v.emplace_back(translated);
    v.emplace_back(rotated);
    v.emplace_back(AffT);
    v.emplace_back(PersT);

    // 定義結果圖片的排列方式 (n 行 m 列)
    int n = 2, m = 3;
    // 初始化最大寬度和高度
    int w = 0, h = 0;
    // 遍歷 vector v 中的所有圖片，找出最大的寬度和高度
    for (const auto& i : v) {
        w = std::max(w, i.cols);
        h = std::max(h, i.rows);
    }
    // 計算組合後圖片的總寬度和總高度
    int combined_width = w * m, combined_height = h * n;
    // 創建一個空白的 Mat 物件，用於儲存組合後的圖片，顏色與原始圖片相同，背景為黑色
    Mat combined(combined_height, combined_width, img.type(), cv::Scalar::all(0));
    // 遍歷所有變形後的圖片，將它們放置到組合後的圖片中
    for (int i = 0; i < 6; ++i) {
        // 計算當前圖片在組合圖片中的行索引和列索引
        int x = i / m, y = i % m;
        // 計算當前圖片在組合圖片中的偏移量
        int off_x = x * h, off_y = y * w;
        // 將當前圖片複製到組合圖片的指定區域
        v[i].copyTo(combined(cv::Rect(off_y, off_x, v[i].cols, v[i].rows)));
    }

    // 顯示組合後的結果圖片，視窗標題為 "Result"
    cv::imshow("Result", combined);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}