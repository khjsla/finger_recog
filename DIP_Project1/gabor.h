/*Gaborfilter 사용 for 방향성 + noise제거 */
#ifndef gabor_h
#define gabor_h

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

Mat gabor(Mat src, vector<pair<float, float>>& vec, int block_size) {
    Mat dst = Mat::zeros(src.rows, src.cols, CV_32F); 
    // 초기화 - 0배열 생성
    //CV_32F(C1): 32 - bit floating - point number : float(-FLT_MAX..FLT_MAX, INF, NAN)

    int size = 15; //kernel 사이즈
    double sigma = 5; //Standard deviation of the gaussian envelope.
    //시그마 값이 중요! kernnel의 -영향범위-너비를 결정합니다.
    double theta = 0; //Orientation of the normal to the parallel stripes of a Gabor function.
    //kernel의 방향성을 결정하는 값. gabor가 추출하는 edge의 방향성을 결정합니다.
    //이 변수의 활용가능성 때문에 - orientation을 구하는데 gabor 필터가 사용되는 것입니다.
    //x 각도의 edge들을 검출 할 수 있게됩니다.
    double lambd = 7; //Wavelength of the sinusoidal factor.
    double gamma = 1; //Spatial aspect ratio. 감마값은 의미가 없습니다. 1로 그냥 설정.
    double psi = 0; //phase offset // CV_PI * 0.5

    int height = src.rows; //img의 height값
    int width = src.cols;
    int index = 0;

    for (int m = 0; m < height; m++){ //m - height
        for (int n = 0; n < width; n++){ //n - width
            if ((m % block_size) == 0 && (n % block_size) == 0) {
                float dx = vec[index].first; 
                float dy = vec[index].second;

                //x거리와 y거리를 사용하여, (dx, dy값) 수평(x)축으로부터의 theta 각도 구하는 방법
                theta = atan2f(dy, dx) + CV_PI / 2;
                //arctan 함수를 사용해서 구한다

                Mat temp;
                Mat gabor = getGaborKernel({ size, size }, sigma, theta, lambd, gamma, psi);
                filter2D(src, temp, CV_32F, gabor);
                //gaborkernel을 src 에 적용시켜줍니다. 그 결과는 temp 입니다.

                int temp_size = block_size - 1; //현재상태, -1
                if (width < n + temp_size) {
                    temp_size = (width - 1) - n;
                }
                if (height < m + block_size - 1 && temp_size >(height - 1) - m) {
                    temp_size = (height - 1) - m;
                }

                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) { //전체적으로 돌려줌
                        if ( m <= i && i <= (m + temp_size) && n <= j && j <= n + temp_size) {
                            dst.at<float>(i, j) = temp.at<float>(i, j);
                        }
                    }
                }
                index++;
            }
        }
    }

    dst.convertTo(dst, CV_8U);

    //BInarization
    //작은 영역별로 thresholding 진행
    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);
    //pplies an adaptive threshold to an array.
    //3. Non-zero value assigned to the pixels for which the condition is satisfied (max값)
    //4. Adaptive thresholding algorithm to use, see AdaptiveThresholdTypes. The BORDER_REPLICATE | BORDER_ISOLATED is used to process boundaries.
    //5. Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV, see ThresholdTypes.    //6. 
    //6. Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    //7. Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.

    return dst;
}

#endif /* gabor_h */
