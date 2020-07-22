#ifndef orientation_h
#define orientation_h

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include "Minutiae.h" 
//for singular 처리 with minutiae header

using namespace std;
using namespace cv;

pair<Mat, vector<pair<float, float>>> orientation(Mat src, Mat d, string name_, int size = 8, bool coredelta = false) {
    Mat inputImage = src;
    Mat draw = d;

    inputImage.convertTo(inputImage, CV_32F, 1.0 / 255, 0);

    medianBlur(inputImage, inputImage, 3); 
    //blur

    int blockSize = size;

    Mat fprintWithDirectionsSmooth = inputImage.clone();
    Mat coredeltaPrint = draw.clone();

    Mat tmp(inputImage.size(), inputImage.type());
    Mat coherence(inputImage.size(), inputImage.type());
    Mat orientationMap = tmp.clone();

    //Gradiants x and y
    Mat grad_x, grad_y;

    Sobel(inputImage, grad_x, inputImage.depth(), 1, 0, 3);
    Sobel(inputImage, grad_y, inputImage.depth(), 0, 1, 3);
    //Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
    //3. output image depth, see combinations; in the case of 8-bit input images it will result in truncated derivatives.
    //4. order of the derivative x.
    //5. order of the derivative y.
    //6. size of the extended Sobel kernel; it must be 1부터 홀수

    Mat Fx(inputImage.size(), inputImage.type()),
        Fy(inputImage.size(), inputImage.type()),
        Fx_gauss,
        Fy_gauss;
    Mat smoothed(inputImage.size(), inputImage.type());

    int width = inputImage.cols;
    int height = inputImage.rows;
    int blockH = 0;
    int blockW;

    vector<pair<float, float>> vec;
    vector<int> cnt;
    
    //blk 반복해서 끝까지
    for (int i = 0; i < height; i += blockSize) { 
        for (int j = 0; j < width; j += blockSize) {
            float Gsx = 0.0; float Gsy = 0.0; float Gxx = 0.0; float Gyy = 0.0;

            //이미지 범위 확인
            blockH = ((height - i) < blockSize) ? (height - i) : blockSize;
            blockW = ((width - j) < blockSize) ? (width - j) : blockSize;

            //WW 블록안의 평균
            //compute gradient Gs each pixel
            //Gsx, Gsy (m,n) 구하기 
            for (int u = i; u < i + blockH; u++) {
                for (int v = j; v < j + blockW; v++) {
                    Gsx += (grad_x.at<float>(u, v)*grad_x.at<float>(u, v)) - (grad_y.at<float>(u, v)*grad_y.at<float>(u, v));
                    Gsy += 2 * grad_x.at<float>(u, v) * grad_y.at<float>(u, v);
                    Gxx += grad_x.at<float>(u, v)*grad_x.at<float>(u, v);
                    Gyy += grad_y.at<float>(u, v)*grad_y.at<float>(u, v);
                }
            }

            //ESTIMATE LOCAL ORIENTATION each blk center at (m,n)
            /**************** 강노에서의 orientation 식 !!!!!!!! ******************/
            float coh = sqrt(pow(Gsx, 2) + pow(Gsy, 2)) / (Gxx + Gyy);
            //smoothed
            float fi = 0.5f * fastAtan2(Gsy, Gsx) * CV_PI / 180.0f;


            Fx.at<float>(i, j) = cos(2 * fi);
            Fy.at<float>(i, j) = sin(2 * fi);

            //blk 채우기
            for (int u = i; u < i + blockH; u++) {
                for (int v = j; v < j + blockW; v++) {
                    orientationMap.at<float>(u, v) = fi;
                    Fx.at<float>(u, v) = Fx.at<float>(i, j);
                    Fy.at<float>(u, v) = Fy.at<float>(i, j);
                    coherence.at<float>(u, v) = (coh < 0.85f) ? 1.0f : 0.0f;
                }
            }
        }
    }

    GaussianBlur(Fx, Fx_gauss, Size(5, 5), 1, 1);
    GaussianBlur(Fy, Fy_gauss, Size(5, 5), 1, 1);
    //blur

    cvtColor(fprintWithDirectionsSmooth, fprintWithDirectionsSmooth, COLOR_GRAY2BGR);

    vector<vector<float>> new_vec(height, vector<float>(width, 0.0f));

    for (int m = 0; m < height; m++) {
        for (int n = 0; n < width; n++) {
            smoothed.at<float>(m, n) = 0.5f * fastAtan2(Fy_gauss.at<float>(m, n), Fx_gauss.at<float>(m, n)) * CV_PI / 180.0f;
            if ((m % blockSize) == 0 && (n % blockSize) == 0) {
                int x = n;
                int y = m;
                int ln = sqrt(2 * pow(blockSize, 2)) / 2;
                float dx = ln * cos(smoothed.at<float>(m, n) - CV_PI / 2.0f);
                float dy = ln * sin(smoothed.at<float>(m, n) - CV_PI / 2.0f);
                vec.push_back({ dx,dy });

                float m = dy / (dx + FLT_EPSILON);

                float mm = m;

                // 4방향 quantazation
                if (2.0f <= mm)
                    mm = FLT_MAX;
                else if (0.5f <= mm && mm < 2.0f)
                    mm = 1.0f;
                else if (-0.5f <= mm && mm < 0.5f)
                    mm = 0.0f;
                else if (-2.0f <= mm && mm < -0.5f)
                    mm = -1.0f;
                else if (mm < -2.0f)
                    mm = FLT_MAX;

                new_vec[m][n] = mm;

                int xx = (blockH / 2) / sqrt(pow(m, 2) + 1);
                int yy = m * xx;

                if (coredelta) {
                    if (mm == 1.0f) {
                        xx = blockSize / 2 - 1;
                        yy = blockSize / 2 - 1;
                    }
                    else if (mm == -1.0f) {
                        xx = blockSize / 2 - 1;
                        yy = -(blockSize / 2 - 1);
                    }
                    else if (mm == 0.0f) {
                        xx = blockSize / 2 - 1;
                        yy = 0;
                    }
                    else if (mm == FLT_MAX) {
                        xx = 0;
                        yy = blockSize / 2 - 1;
                    }
                }

                if (xx == 0 && yy == 0)
                    yy = blockH / 2;

                if (!coredelta)
                    line(fprintWithDirectionsSmooth, Point(x, y + blockH), Point(x + dx, y + blockH + dy), Scalar(0, 0, 255), 1, LINE_AA);
            }
        }
    }

    priority_queue<pair<int, pair<int, int>>> core_priorityQ; //pq 사용
    priority_queue<pair<int, pair<int, int>>> delta_priorityQ;

    /***********         SINGULAR POINT 구하기       *************/
    if (coredelta) { //coredelta값이면
        for (int m = 0; m < height; m++) {
            for (int n = 0; n < width; n++) {
                if (m % blockSize == 0 && n % blockSize == 0) {
                    if (0 <= m - blockSize && m + blockSize < height &&
                        0 <= n - blockSize && n + blockSize < width) {
                        // 왼쪽 중간: +, 중간 중간: =, 오른쪽 중간: - 인 경우, 위 아래에서 각각 가로, 세로 값 찾기
                        if (new_vec[m][n - blockSize] == -1.0f && new_vec[m][n] == 0.0f && new_vec[m][n + blockSize] == 1.0f) {
                            int up_up = 0;
                            int up_side = 0;
                            int down_up = 0;
                            int down_side = 0;

                            // 북서
                            if (new_vec[m - blockSize][n - blockSize] == FLT_MAX)
                                up_up += 2;
                            else if (new_vec[m - blockSize][n - blockSize] == 1.0f || new_vec[m - blockSize][n - blockSize] == -1.0f) {
                                up_up++;
                                up_side++;
                            }
                            else if (new_vec[m - blockSize][n - blockSize] == 0.0f)
                                up_side += 2;

                            // 북
                            if (new_vec[m - blockSize][n] == FLT_MAX)
                                up_up += 2;
                            else if (new_vec[m - blockSize][n] == 1.0f || new_vec[m - blockSize][n] == -1.0f) {
                                up_up++;
                                up_side++;
                            }
                            else if (new_vec[m - blockSize][n] == 0.0f)
                                up_side += 2;

                            // 북동
                            if (new_vec[m - blockSize][n + blockSize] == FLT_MAX)
                                up_up += 2;
                            else if (new_vec[m - blockSize][n + blockSize] == 1.0f || new_vec[m - blockSize][n + blockSize] == -1.0f) {
                                up_up++;
                                up_side++;
                            }
                            else if (new_vec[m - blockSize][n + blockSize] == 0.0f)
                                up_side += 2;


                            // 남서
                            if (new_vec[m + blockSize][n - blockSize] == FLT_MAX)
                                down_up += 2;
                            else if (new_vec[m + blockSize][n - blockSize] == 1.0f || new_vec[m + blockSize][n - blockSize] == -1.0f) {
                                down_up++;
                                down_side++;
                            }
                            else if (new_vec[m + blockSize][n - blockSize] == 0.0f)
                                down_side += 2;

                            // 남
                            if (new_vec[m + blockSize][n] == FLT_MAX)
                                down_up += 2;
                            else if (new_vec[m + blockSize][n] == 1.0f || new_vec[m + blockSize][n] == -1.0f) {
                                down_up++;
                                down_side++;
                            }
                            else if (new_vec[m + blockSize][n] == 0.0f)
                                down_side += 2;

                            // 북동
                            if (new_vec[m + blockSize][n + blockSize] == FLT_MAX)
                                down_up += 2;
                            else if (new_vec[m + blockSize][n + blockSize] == 1.0f || new_vec[m + blockSize][n + blockSize] == -1.0f) {
                                down_up++;
                                down_side++;
                            }
                            else if (new_vec[m + blockSize][n + blockSize] == 0.0f)
                                down_side += 2;

                            int cnt_core = up_side + down_up;
                            int cnt_delta = up_up + down_side;

                            if (abs(cnt_delta - cnt_core) > 2) { //절댓값이 3이상이면 고려
                                if (cnt_core >= cnt_delta) //core>면 core
                                    core_priorityQ.push({ cnt_core - cnt_delta, {m, n} });
                                else                       //아니면 delta
                                    delta_priorityQ.push({ cnt_delta - cnt_core, {m, n} });
                            }
                        }
                    }
                }
            }
        }
    }
    else { //coredelta의 경우가 아니면        
        int my_index = 0;
        for (int m = 0; m < height; m++) {
            for (int n = 0; n < width; n++) {
                if (m % blockSize == 0 && n % blockSize == 0) {
                    if (0 <= m - blockSize && m + blockSize < height) { // 위 아래가 | 방향인 경우
                        if (new_vec[m - blockSize][n] == FLT_MAX && new_vec[m + blockSize][n] == FLT_MAX) { // 현재 방향이 ㅡ 
                            if (new_vec[m][n] == 1.0f || new_vec[m][n] == -1.0f || new_vec[m][n] == 0.0f) { // 방향이 다르면 |로 넣어줌
                                if (vec[my_index - width / blockSize].second * vec[my_index + width / blockSize].second < 0.0f)
                                    vec[my_index] = { 0.0f, FLT_MAX };  // 방향이 같은 경우 두 값의 평균으로 값을 채워 넣음으로써 흉터를 제할 수 있습니다.
                                else {
                                    vec[my_index] = {
                                        (vec[my_index - width / blockSize].first + vec[my_index + width / blockSize].first) / 2 ,
                                        (vec[my_index - width / blockSize].second + vec[my_index + width / blockSize].second) / 2
                                    };
                                }
                            }
                        }
                    }

                    my_index++;
                }
            }
        }
    }

    int blan = 0;
    int co = 10; //core는 type 10 
    int del = 11; //delta는 type 11
    char nn[101] = "/Users/kimdayeong/Downloads/Fingerprint_DB_(40)/bin/";
    string name = name_ + ".bin";
    FILE* file = fopen(strcat(nn, name.c_str()), "a");
    int n = core_priorityQ.size() + delta_priorityQ.size();
    cout << "\n num: "<< n<< endl;
    print_a(name_, n, true);
    
    // core는 네모
    if (!core_priorityQ.empty()) {
        rectangle(coredeltaPrint, Point(core_priorityQ.top().second.second + blockSize / 2 - 5, core_priorityQ.top().second.first + blockSize / 2 - 5), Point(core_priorityQ.top().second.second + blockSize / 2 + 5, core_priorityQ.top().second.first + blockSize / 2 + 5), Scalar(0, 0, 255));
        int x = core_priorityQ.top().second.second + blockSize / 2;
        int y = core_priorityQ.top().second.first + blockSize / 2;

        //bin file에 입력
        fwrite(&x, sizeof(int), 1, file);
        fwrite(&y, sizeof(int), 1, file);
        fwrite(&blan, sizeof(unsigned char), 1, file);
        fwrite(&co, sizeof(unsigned char), 1, file);
    }
    //Singular point 를 찾는다.

    // delta
    if (!delta_priorityQ.empty()) {
        int mmm = delta_priorityQ.top().second.first;
        int nnn = delta_priorityQ.top().second.second;

        line(coredeltaPrint, Point(nnn + blockSize / 2, mmm), Point(nnn + blockSize / 2, mmm + blockSize), Scalar(0, 0, 255), 1, LINE_AA, 0);
        line(coredeltaPrint, Point(nnn, mmm + blockSize / 2), Point(nnn + blockSize, mmm + blockSize / 2), Scalar(0, 0, 255), 1, LINE_AA, 0);
        delta_priorityQ.pop();
    }
    if (!delta_priorityQ.empty()) {
        int mmm = delta_priorityQ.top().second.first;
        int nnn = delta_priorityQ.top().second.second;
        line(coredeltaPrint, Point(nnn + blockSize / 2, mmm), Point(nnn + blockSize / 2, mmm + blockSize), Scalar(0, 0, 255), 1, LINE_AA, 0);
        line(coredeltaPrint, Point(nnn, mmm + blockSize / 2), Point(nnn + blockSize, mmm + blockSize / 2), Scalar(0, 0, 255), 1, LINE_AA, 0);
        int x = core_priorityQ.top().second.second + blockSize / 2;
        int y = core_priorityQ.top().second.first + blockSize / 2;
        
        //bin file에 입력!
        fwrite(&x, sizeof(int), 1, file);
        fwrite(&y, sizeof(int), 1, file);
        fwrite(&blan, sizeof(unsigned char), 1, file);
        fwrite(&del, sizeof(unsigned char), 1, file);
    }


    normalize(orientationMap, orientationMap, 0, 1, NORM_MINMAX);
    //Normalizes the norm or value range of an array.
    //3. alpha – norm value to normalize to or the lower range boundary in case of the range normalization.
    //4. beta – upper range boundary in case of the range normalization; it is not used for the norm normalization.
    //5. normType – normalization type (see the details below).
    //when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be normalized. 
    //This means that the norm or min-n-max are calculated over the sub-array, and then this sub-array is modified to be normalized.

    orientationMap = smoothed.clone();
    normalize(smoothed, smoothed, 0, 1, NORM_MINMAX);
    //위랑 같음

    pair<Mat, vector<pair<float, float>>> returning;
    if (coredelta)
        returning = { coredeltaPrint, vec };
    else
        returning = { fprintWithDirectionsSmooth, vec };

    return returning;
}

#endif /* orientation_h */
