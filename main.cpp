#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;

void ElasticDeformations(cv::Mat& src,
                         cv::Mat& dst,
                         bool bNorm = false,
                         double sigma = 4.,
                         double alpha = 34.)
{
    if(src.empty()) return;
    if(dst.empty()) dst = cv::Mat::zeros(src.size(), src.type());

    cv::Mat dx(src.size(), CV_64FC1);
    cv::Mat dy(src.size(), CV_64FC1);

    double low = -1.0;
    double high = 1.0;

    //The image deformations were created by first generating
    //random displacement fields, that's dx(x,y) = rand(-1, +1) and dy(x,y) = rand(-1, +1)
    cv::randu(dx, cv::Scalar(low), cv::Scalar(high));
    cv::randu(dy, cv::Scalar(low), cv::Scalar(high));

    //The fields dx and dy are then convolved with a Gaussian of standard deviation sigma(in pixels)
    cv::Size kernel_size(sigma*6 + 1, sigma*6 + 1);
    cv::GaussianBlur(dx, dx, kernel_size, sigma, sigma);
    cv::GaussianBlur(dy, dy, kernel_size, sigma, sigma);

    //If we normalize the displacement field (to a norm of 1,
    //the field is then close to constant, with a random direction
    if(bNorm)
    {
        dx /= cv::norm(dx, cv::NORM_L1);
        dy /= cv::norm(dy, cv::NORM_L1);
    }

    //The displacement fields are then multiplied by a scaling factor alpha
    //that controls the intensity of the deformation.
    dx *= alpha;
    dy *= alpha;

    //Inverse(or Backward) Mapping to avoid gaps and overlaps.
    cv::Rect checkError(0, 0, src.cols, src.rows);
    int nCh = src.channels();

    for(int displaced_y = 0; displaced_y < src.rows; displaced_y++)
        for(int displaced_x = 0; displaced_x < src.cols; displaced_x++)
        {
            int org_x = displaced_x - dx.at<double>(displaced_y, displaced_x);
            int org_y = displaced_y - dy.at<double>(displaced_y, displaced_x);

            if(checkError.contains(cv::Point(org_x, org_y)))
            {
                for(int ch = 0; ch < nCh; ch++)
                {
                    dst.data[(displaced_y * src.cols + displaced_x) * nCh + ch] = src.data[(org_y * src.cols + org_x) * nCh + ch];
                }
            }
        }
}


int main()
{
    cv::Mat src = cv::imread("mnist_input_0.png",cv::IMREAD_GRAYSCALE);

    while(true)
    {
        cv::Mat dst_not_norm, dst_norm;

        ElasticDeformations(src,dst_not_norm);
        ElasticDeformations(src,dst_norm, true);

        cv::imshow("src", src);
        cv::imshow("dst_not_norm", dst_not_norm);
        cv::imshow("dst_norm", dst_norm);

        //you pushed any key, image is updated.
        cv::waitKey(0);
    }
    return 0;
}

