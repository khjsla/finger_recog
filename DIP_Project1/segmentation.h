#ifndef segmentation_h
#define segmentation_h

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

Mat segmentation(Mat &src, Mat &dst) {
    src.convertTo(dst, CV_8UC1);
    pyrUp(dst, dst, Size(1 / 4, 1 / 4)); //사이즈 변경

    medianBlur(dst, dst, 7);
    //The median filter uses BORDER_REPLICATE internally to cope with border pixels, see BorderTypes
    //1. input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
    //2. destination array of the same size and type as src.
    //3. aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
    
    Mat mask = getStructuringElement(1, Size(3, 3), Point(1, 1));
    //Returns a structuring element of the specified size and shape for morphological operations.
    //The function constructsand returns the structuring element that can be further passed to erode, dilate or morphologyEx.But you can also construct an arbitrary binary mask yourself and use it as the structuring element
    //adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 2);
    //1. Element SHAPE that could be one of MorphShapes
    //2. Size of the structuring element.
    //3. Anchor position within the element. The default value (−1,−1) means that the anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor position. In other cases the anchor just regulates how much the result of the morphological operation is shifted.

    Mat segmentation;
    morphologyEx(dst, segmentation, MORPH_OPEN, mask, Point(-1, -1), 12);
    //Performs advanced morphological transformations
    //The function cv::morphologyEx can perform advanced morphological transformations using an erosionand dilation as basic operations.
    //Any of the operations can be done in - place.In case of multi - channel images, each channel is processed independently.
    //3. TYPE of a morphological operation, see MorphTypes
    //4. Structuring element. It can be created using getStructuringElement.
    //5. ANCHOR POSITION with the kernel. Negative values mean that the anchor is at the kernel center.
    
    GaussianBlur(dst, dst, Size(9, 9), 7);
    //Blurs an image using a Gaussian filter.
    //The function convolves the source image with the specified Gaussian kernel.In - place filtering is supported.
    //3. Gaussian KERNEL SIZE. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma.
    //4. Gaussian kernel standard deviation in X direction.
    //5. y생략시 x와 같은 값으로 적용됨.

    adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 1); 

    erode(dst, dst, mask, Point(-1, -1), 1); 
    //erode가 됩니다.
    //Erodes an image by using a specific structuring element.
    //The function erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken
    //3. structuring element used for erosion; if element=Mat(), a 3 x 3 rectangular structuring element is used. Kernel can be created using getStructuringElement.
    //4. position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center. (anchor)
    //5. number of times erosion is applied. (interation)

    pyrDown(dst, dst, Size(1 / 4, 1 / 4)); //사이즈 변경
    pyrDown(segmentation, segmentation, Size(1 / 4, 1 / 4));

    return segmentation;
}
#endif /* segmentation_h */
