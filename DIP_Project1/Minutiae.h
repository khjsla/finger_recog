#ifndef Minutiae_h
#define Minutiae_h
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>

using namespace std;
using namespace cv;


struct Minutiae {
    int x;
    int y;
    int angle;
    int type; //ending:1  bifurcation:2
};

vector<Minutiae> findMinutiae(Mat& img, Mat& seg) {
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);

    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);


    Mat area;
    seg.convertTo(area, CV_8UC1);
    Mat mask = getStructuringElement(1, Size(3, 3), Point(1, 1));
    dilate(area, area, mask, Point(-1, -1), 7);

    int ending = 0;
    int bifurcation = 0;
    vector<Minutiae> mVector;
    Minutiae minutiae;

    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int x, y;
    uchar* pUp; uchar* pCurr;  uchar* pDown;
    uchar *northWest, *north, *northEast;    // north (pUp)
    uchar *west, *median, *east;
    uchar *southWest, *south, *southEast;    // south (pDown)

    uchar *pDst;

    pUp = NULL;
    pCurr = img.ptr<uchar>(0);
    pDown = img.ptr<uchar>(1);

    // minutiae 개수 조절 - filtering
    for (int thr = 5; thr < 30; thr++) {
        mVector.clear();
        ending = 0;
        bifurcation = 0;

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
                northWest = north;   north = northEast;   northEast = &(pUp[x + 1]);
                west = median;   median = east;   east = &(pCurr[x + 1]);
                southWest = south;   south = southEast;   southEast = &(pDown[x + 1]); 
                //하나씩 오른쪽으로 옮겨짐

                //ending 구하기용
                //ending이면 median + 주위 8개중 1개가 1 임
                int sum = *northWest + *north + *northEast + *west + *east + *southWest + *south + *southEast;
                //sum으로 1인지 확인하면 end 찾을 수 있음

                //bif 구하기용
                //xor 기본 식이 있어서 xor_로 변수 설정
                //각 칸의 xor관계식을 구할수 있음
                int xor_ = (*northWest ^ *north) + (*north ^ *northEast) + (*northEast ^ *east) + (*east ^ *southEast)
                    + (*west ^ *southWest) + (*southWest ^ *south) + (*south ^ *southEast) + (*west ^ *northWest);
                //번갈아서 칸이 채워지게 되면 bif로 판단

                if (*median == 1 && (sum == 1)) { //앞서 설명한 ending의 조건
                    uchar* segVal = &(area.ptr<uchar>(y))[x];
                    if (*segVal == 0) {
                        bool isalready = false;
                        // 현재 찾은 Minutiae를 모두 찾아 - 근처인지 확인
                        for (auto mnt : mVector) {
                            int distt = sqrt(pow(mnt.x - x, 2) + pow(mnt.y - y, 2)); 
                            //거리공식 으로 구해서
                            if (distt <= thr) { //그 거리가 threshold값보다 작으면
                                isalready = true;  //이미 측정한 것으로 판단하고,
                                //방금 구한 점을 제외할수 있게된다
                                break;
                            }
                        }
                        if (!isalready) {
                            ending++; ///개수
                            minutiae.x = x; 
                            minutiae.y = y;
                            minutiae.type = 1;
                            //하나의 minutiae 만듦
                            mVector.push_back(minutiae);
                        }
                    }
                }

                if (*median == 1 && xor_ == 6) { //앞서 설명한 bifu의 조건
                    uchar* segVal = &(area.ptr<uchar>(y))[x];
                    if (*segVal == 0) {
                        bool isalready = false;
                        // minutiae 근처에 있는지 확인
                        for (auto mnt : mVector) {
                            int distt = sqrt(pow(mnt.x - x, 2) + pow(mnt.y - y, 2));
                            //거리공식
                            //위랑 같은 mechani
                            if (distt <= thr) {
                                isalready = true; 
                                break;
                            }
                        }
                        if (!isalready) {
                            bifurcation++; //개수
                            minutiae.x = x; 
                            minutiae.y = y;
                            minutiae.type = 2; 
                            //하나의 minutiae 만듦
                            mVector.push_back(minutiae);
                        }
                    }
                }
            }
        }

        // 개수 줄이기 위한 threshold 바꾸기
        if (ending <= 30 && bifurcation <= 30) {
            cout << "thr: " << thr << endl;
            break;
        }
    }
    cout << "end: " << ending << ", bif: " << bifurcation << endl;

    return mVector;
}


float angle(Mat& dst, vector<pair<float, float>>& vec, int& u, int& v, int& block_size, Size size, int& type) {
    float fi = 0.0;

    int val = size.width / block_size;
    int width = u / block_size;
    int height = v / block_size;
    
    //각도 구하는 부분!!!
    //두 선분의 길이를 알고있을때 그 사이각 - 세타 값을 구하기 위해선
    //arctan
    fi = -atan2f(vec[height*val + width].second, vec[height*val + width].first) * 180 / CV_PI;

    // end
    if (type == 1) {
        //  각도가 오른쪽 위
        if (fi > 0) {
            // 왼쪽 아래에 다음 있으면 -180
            if (dst.at<uchar>({ u - 1, v }) == 1 || dst.at<uchar>({ u - 1, v + 1 }) == 1 || dst.at<uchar>({ u, v + 1 }) == 1)
                fi -= 180;
        }
        // 각도가 오른쪽 아래
        else if (fi < 0) {
            // 오른쪽 위 다음 있으면 +180
            if (dst.at<uchar>({ u - 1, v }) == 1 || dst.at<uchar>({ u - 1, v - 1 }) == 1 || dst.at<uchar>({ u, v - 1 }) == 1)
                fi += 180;
        }
    }
    // bifar
    else if (type == 2) {
        // 방문 저장
        vector<vector<bool>> visit(size.width, vector<bool>(size.height, false));
        visit[u][v] = true;
        int dir = 0;

        // 기준점 3방향 나누기
        pair<int, int> index[3] = { { -1,-1 }, {-1,-1}, {-1,-1} };

        // t = 0 일때 상하좌우 부터 찾고, 3개 다 못찾으면 대각선 방향 찾음
        for (int t = 0; t < 2; t++) {
            for (int southEast = -1; southEast <= +1; southEast++) {
                for (int j = -1; j <= +1; j++) {
                    if (t == 1 || (t == 0 && southEast*j == 0)) {
                        if (0 <= u + southEast && u + southEast < size.width && 0 <= v + j && v + j < size.height &&
                            !visit[u + southEast][v + j] && dst.at<uchar>({ u + southEast, v + j }) == 1) {
                            if (dir < 3) {
                                index[dir++] = { u + southEast, v + j };
                                visit[u + southEast][v + j] = true;
                            }
                        }
                    }
                }
            }
        }

        // 이동할 횟수
        int count = 10;
        for (int tt = 0; tt < count; tt++) {
            for (int dir = 0; dir < 3; dir++) {
                bool isfinish = false;
                // t = 0 일때 상하좌우 부터 탐색
                for (int t = 0; t < 2; t++) {
                    //탐색
                    for (int southEast = -1; southEast <= +1; southEast++) {
                        for (int j = -1; j <= +1; j++) {
                            if (t == 1 || (t == 0 && southEast * j == 0)) {
                                int next_i = index[dir].first + southEast;
                                int next_j = index[dir].second + j;

                                // 방향, 점 있으면
                                if (0 <= next_i && next_i < size.width &&
                                    0 <= next_j && next_j < size.height &&
                                    !visit[next_i][next_j] &&
                                    dst.at<uchar>({ next_i, next_j }) == 1) {
                                    index[dir] = { next_i, next_j };
                                    visit[next_i][next_j] = true;

                                    isfinish = true;
                                    break;
                                }
                            }
                        }
                        if (isfinish)
                            break;
                    }
                    if (isfinish)
                        break;
                }
            }
        }

        // 3 개 중 2개를 골라 각도가 제일 작은 것을 저장
        int min_theta = 361;
        int min_one = -1, min_two = -1;
        for (int southEast = 0; southEast < 3; southEast++) {
            int one = -1, two = -1;
            // 2개를 고름
            for (int j = 0; j < 3; j++) {
                if (southEast == j)
                    continue;
                if (one == -1)
                    one = j;
                else
                    two = j;
            }
            // 2개의 벡터를 구함
            int v1x = index[one].first - u;
            int v1y = index[one].second - v;
            int v2x = index[two].first - u;
            int v2y = index[two].second - v;

            // 두 벡터의 내적을 구함
            float inner = v1x * v2x + v1y * v2y;
            // 두 벡터의 크기를 구함
            float v1_size = sqrt(pow(v1x, 2) + pow(v1y, 2));
            float v2_size = sqrt(pow(v2x, 2) + pow(v2y, 2));

            // 두 벡터의 사이 각을 구함
            int theta = acosf(inner / (v1_size*v2_size)) * 180.0f / CV_PI;

            // 최소값보다 작으면 갱신
            if (min_theta > theta) {
                min_theta = theta;
                min_one = one;
                min_two = two;
            }
        }

        // 가장 작은 각도를 가진 두벡터의 중간 값을 구함
        float mid_x = (index[min_one].first + index[min_two].first) / 2.0f;
        float mid_y = (index[min_one].second + index[min_two].second) / 2.0f;

        // 현재 점에서 중간 값으로의 벡터를 구함
        float vx = mid_x - u;
        float vy = mid_y - v;

        // 방향이 같은 경우 각도를 180도 바꿔줌
        if (0 < fi) {
            if (0 < vx && 0 < vy)
                fi -= 180;
        }
        else if (fi < 0) {
            if (vx < 0 && vy < 0)
                fi += 180;
        }
    }

    return fi;
}

vector<Minutiae> mVector;


Mat printMinutiae(Mat src, Mat& srcc, vector<pair<float, float>>& vec, int& block_size, Size size, Mat& original) {
    Mat temp;
    Mat dst = src.clone();
    dst /= 255;         // convert to binary image

    Mat dst2 = original.clone();
    cvtColor(dst2, dst2, COLOR_GRAY2BGR);

    mVector = findMinutiae(dst, srcc);


    for (int southEast = 0; southEast < mVector.size(); southEast++)
        mVector[southEast].angle = angle(dst, vec, mVector[southEast].x, mVector[southEast].y, block_size, size, mVector[southEast].type);

    dst *= 255;
    cvtColor(dst, dst, COLOR_GRAY2RGB);
    threshold(dst, dst, 127, 255, THRESH_BINARY_INV);

    Scalar end = Scalar(0, 255, 255);
    Scalar bif = Scalar(255, 255, 0);
    int num = 0;

    //cvtColor(original, original, COLOR_GRAY2BGR);
    for (int southEast = 0; southEast < mVector.size(); southEast++) {
        if (mVector[southEast].type == 1) {
            circle(dst2, Point(mVector[southEast].x, mVector[southEast].y), 3, end, 1, 8);
            line(dst2, { mVector[southEast].x, mVector[southEast].y },
                { mVector[southEast].x + (int)(8.0f * cos(-mVector[southEast].angle * CV_PI / 180.0f)), mVector[southEast].y + int(8.0f * sin(-mVector[southEast].angle * CV_PI / 180.0f)) }
            , end);
            num++;
        }
        else if (mVector[southEast].type == 2) {
            circle(dst2, Point(mVector[southEast].x, mVector[southEast].y), 3, bif, 1, 8);
            line(dst2, { mVector[southEast].x, mVector[southEast].y },
                { mVector[southEast].x + (int)(8.0f * cos(-mVector[southEast].angle * CV_PI / 180.0f)), mVector[southEast].y + int(8.0f * sin(-mVector[southEast].angle* CV_PI / 180.0f)) }
            , bif);
            num++;
        }
    }
    return dst2;
    //return mVector<Minutiae>;
}

int end_ = 1;
int bif_ = 3;

//bin file만들기! with orientation.h 의 singular point들
int print_a(string name_, int n = 0, bool p = false) {
    if(p == true) {
            string name = name_ + ".bin";
        //cout << name;
        //char aa[512] = {};
        //name.toCharArray(aa, name.length());
        //sprintf(aa, "%s", name);
        char nn[101] = "/Users/kimdayeong/Downloads/Fingerprint_DB_(40)/bin/";
            FILE* file = fopen(strcat(nn, name.c_str()), "w+");
            int num = 0;
            int wid = 155; int hei = 200;
            fwrite(&wid, sizeof(int), 1, file);
            fwrite(&hei, sizeof(int), 1, file);
            for (int southEast = 0; southEast < mVector.size(); southEast++) {
                if (mVector[southEast].type == 1) num++;
                else if (mVector[southEast].type == 2) num++;
            }
    
            fwrite(&num, sizeof(int), 1, file);
            fwrite(&n, sizeof(int), 1, file);
            for (int southEast = 0; southEast < mVector.size(); southEast++) {
                if (mVector[southEast].type == 1) {
                    unsigned char ang = (mVector[southEast].angle + 180) * 255 / 360;
                    cout<<"x: "<<mVector[southEast].x<<' '<<" y: "<<mVector[southEast].y<<" theta: "<<(mVector[southEast].angle + 180)<<" type: "<<mVector[southEast].type<<"\n";
    
                    fwrite(&mVector[southEast].x, sizeof(int), 1, file);
                    fwrite(&mVector[southEast].y, sizeof(int), 1, file);
                    fwrite(&ang, sizeof(unsigned char), 1, file);
                    fwrite(&end_, sizeof(unsigned char), 1, file);
    
                }
                else if (mVector[southEast].type == 2) {
                    unsigned char ang = (mVector[southEast].angle + 180) * 255 / 360;
                    cout<<"x: "<<mVector[southEast].x<<' '<<" y: "<<mVector[southEast].y<<" theta: "<<(mVector[southEast].angle + 180)<<" type: "<<mVector[southEast].type<<"\n";
    
                    fwrite(&mVector[southEast].x, sizeof(int), 1, file);
                    fwrite(&mVector[southEast].y, sizeof(int), 1, file);
                    fwrite(&ang, sizeof(unsigned char), 1, file);
                    fwrite(&bif_, sizeof(unsigned char), 1, file);
                }
            }
            fclose(file);
        }
    return 0;
}

#endif /* Minutiae_h */
