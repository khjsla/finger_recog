//  main.cpp
//  Project1
//
//  Created by 김다영 on 2020/06/06.
//  Copyright © 2020 김다영. All rights reserved.
//

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> //BIn file for mac os
#include <stdio.h>

#include "gabor.h"
#include "orientation.h"
#include "segmentation.h"
#include "thinning.h"
#include "Minutiae.h"

using namespace std;
using namespace cv;

int main() {
    // orientation block size
    // rows, cols must be devided by block size
    int block_size = 7;

    string names[101] = { //파일들의 이름
        "[10_G]_2019_5_1_L_I#2",
        "[23]_2019_8_4_L_T_1",
        "[36]_2019_5_1_L_I#1",
        "[11_Im]_2019_2_1_R_색_2",
        "[24]_2019_8_3_L_M_1",
        "[37_G]_2019_5_1_L_M#2",
        "[12]_2019_2_1_L_약_1",
        "[25_Im]_2019_8_3_R_P_1",
        "[38_Im]_2019_5_1_R_P#1",
        "[13]_2019_2_1_L_엄_1",
        "[26_Im]_2019_8_3_R_T_2",
        "[39_G]_2019_5_1_L_R#2",
        "[14_G]_2019_2_1_L_중_2",
        "[27_G]_2019_5_2_L_I_2",
        "[3_Im]_2019_8_3_R_I_2",
        "[15_Im]_2019_8_1_R_I_1",
        "[28_Im]_2019_5_2_R_M_1",
        "[40_G]_2019_5_1_L_T#2",
        "[16_Im]_2019_8_1_R_M_1",
        "[29_Im]_2019_5_2_R_P_2",
        "[4]_2019_8_2_L_I_1",
        "[17]_2019_8_1_L_P_1",
        "[2_G]_2019_8_1_L_I_2",
        "[5]_2019_2_1_L_검_1",
        "[18]_2019_8_1_L_R_1",
        "[30_Im]_2019_5_2_R_T_2",
        "[6_G]_2019_2_1_L_검_2",
        "[19_G]_2019_8_1_L_T_2",
        "[31_Im]_2019_5_3_R_I_1",
        "[7_G]_2019_2_1_L_검_3",
        "[1_Im]_2019_8_4_R_I_1",
        "[32_Im]_2019_5_3_R_M_1",
        "[8_Im]_2019_2_1_R_검_1",
        "[20_Im]_2019_8_4_R_M_1",
        "[33_G]_2019_5_3_L_P_2",
        "[9]_2019_5_4_L_I_1",
        "[21_Im]_2019_8_4_R_P_1",
        "[34_Im]_2019_5_3_R_R_1",
        "[22_G]_2019_8_4_L_R_2",
        "[35_Im]_2019_5_3_R_T_1"
    };

    for(int i = 0; i < 40; i++) {
        string name = "finger/" + names[i] + ".bmp"; 
        //파일 이름으로 bin파일 생성하기 위해 
        Mat src = imread(name);
        Size size = { 152,200 }; //resize로 크기설정해줄것 
        cvtColor(src, src, COLOR_RGB2GRAY);
        Mat orig = src.clone(); //이미지복사

    // resize
        resize(src, src, size);

        Mat segmented;
    // segmantation image
        Mat segmented2 = segmentation(src, segmented);
    
    /* Intensity Normalize image */
        equalizeHist(src, src);
        pyrUp(src, src);
        imshow("Normalized", src);
        pyrDown(src, src);

    // block orient
        pair<Mat, vector<pair<float, float>>> returned = orientation(src, orig, names[i], block_size);
        Mat show = returned.first;
        vector<pair<float, float>> vec = returned.second;

    // gabor filter
        Mat gabored = gabor(src, vec, block_size) + segmented2;

        Mat gabored_end;
    // binarization
        threshold(gabored, gabored_end, 1, 255, THRESH_BINARY_INV);

    // thinning
        Mat imgt = thinning(gabored_end);

    // find minutiae and visual them
        Mat result = printMinutiae(imgt, segmented2, vec, block_size, size, orig);
        Mat temp_;
        threshold(segmented2, temp_, 240, 255, THRESH_BINARY);//회색(그림자) 지우기

     //단색 이미지만 가능, 주의! 입력된 영상(temp)도 변경된다.
     //윤곽선 draw
        vector<vector<Point> > contours; //외곽선 배열
        vector<Vec4i> hierarchy;//외곽선들 간의 계층구조
        temp_ = 255 - temp_;
        findContours(temp_, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        int nAreaCnt = contours.size();
        for (int i = 0; i < nAreaCnt; i++) {
            Scalar color = Scalar(0, 204, 0);//윤곽색
            int thickness = 1;//CV_FILLED=내부 채움
         //drawContours(result, contours, i, color, thickness, 8, hierarchy);//외곽선 그리기
        }
    
        pair<Mat, vector<pair<float, float>>> returned_ = orientation(src, result, names[i], 7, true);
        Mat coredelta = returned_.first;
    
    imshow("Input", orig);

    imshow("2.Normalized", src);

    imshow("3.Block orientation", show);

    imshow("1.Segmentation", segmented2);

    gabored.convertTo(gabored, CV_8U);
    //CV_8UC1: 8bit unsigned integer : uchar(0..255)
    //U - unsigned 의미

    imshow("4.Gabor Filtering", gabored);

    imgt.convertTo(imgt, CV_8U);
    imshow("5.Thinning", imgt);

    imshow("6.Minuatiae", result);

    imshow("Result", coredelta);

    waitKey(0);
    }
    return 0;
}
