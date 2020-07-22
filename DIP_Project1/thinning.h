#ifndef thinning_h
#define thinning_h

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

static void thin(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()==true) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar* pUp; uchar* pCurr; uchar* pDown;
    uchar* northWest, * north, * northEast;    // 북서쪽, 북쪽, 북동쪽 (위)
    uchar* west, * median, * east;             // 서쪽 , 중간, 동쪽
    uchar* southWest, * south, * southEast;    // 남서쪽, 남쪽, 동남쪽 (아래)
    uchar *pDst;

    //행(ㅡ) 포인터 초기화
    pUp = NULL;
    pCurr = img.ptr<uchar>(0);
    pDown = img.ptr<uchar>(1);

    for (y = 1; y < img.rows - 1; ++y) {
        //한줄 올림
        pUp = pCurr;
        pCurr = pDown;
        pDown = img.ptr<uchar>(y + 1);
        pDst = marker.ptr<uchar>(y);
        
        //열(ㅣ) 포인터 초기화
        north = &(pUp[0]);
        northEast = &(pUp[1]);
        median = &(pCurr[0]);
        east = &(pCurr[1]);
        south = &(pDown[0]);
        southEast = &(pDown[1]);

        for (x = 1; x < img.cols - 1; ++x) { 
            //열(ㅣ) 포인터를 하나씩 왼쪽으로 이동 - 좌우 확인
            northWest = north;
            north = northEast;
            northEast = &(pUp[x + 1]);
            west = median;
            median = east;
            east = &(pCurr[x + 1]);
            southWest = south;
            south = southEast;
            southEast = &(pDown[x + 1]);

            //검은 check
            int A = (*north == 0 && *northEast == 1) + (*northEast == 0 && *east == 1) +
                (*east == 0 && *southEast == 1) + (*southEast == 0 && *south == 1) +
                (*south == 0 && *southWest == 1) + (*southWest == 0 && *west == 1) +
                (*west == 0 && *northWest == 1) + (*northWest == 0 && *north == 1);
            int B = *north + *northEast + *east + *southEast + *south + *southWest + *west + *northWest;
            int m1 = iter == 0 ? (*north * *east * *south) : (*north * *east * *west);
            int m2 = iter == 0 ? (*east * *south * *west) : (*north * *south * *west);
            
            //med의 검은 이웃의 수가 2에서6이면
            //the number of black pixel neighbours of [median] is between 2-6
            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
                pDst[x] = 1; //값 1로 설정
                //the number of transitions from white to black of [median] is 1
            }
        }
    }
    img &= ~marker;
}


Mat thinning(const cv::Mat& src)
{
    Mat dst;
    dst = src.clone();
    dst /= 255; 
    //binaryimg로 바꾸기! (0_255)  
    //0 = white, 127 = gray, 255 = black

    Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1); //0행렬
    Mat diff;

    //thinning 을 합니다.
    do {
        thin(dst, 0);//위 thin실행
        thin(dst, 1);

        absdiff(dst, prev, diff);
        //Calculates the per-element absolute difference between two arrays or between an array and a scalar.
        //1. first input array or a scalar.
        //2. second input array or a scalar.
        //3. single input array.
        dst.copyTo(prev);

    } while (cv::countNonZero(diff) > 0); //최소 하나이상 (0초과)  [north] and [east] and [south] is 백
    //diff - src – single-channel array. 의 0아닌 array elem cnt 

    dst *= 255; //255를 곱해줍니다.

    return dst;
}

#endif /* thinning_h */
