#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>
#include <vector>

// https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/?utm_source=blog&utm_medium=18_open-Source_computer_vision_projects

int main(){

    std::string path = "C:/Users/Kadir/Desktop/OpenCVProjects/LaneDetection/src/frames/"; // 270, 480
    cv::Mat img; 

    clock_t start, end;

    std::vector<cv::Point> roi_points = {cv::Point(50,270), cv::Point(220,160), cv::Point(360,160), cv::Point(480,270)};

    cv::Mat mask(270, 480, CV_8UC3, cv::Scalar(0, 0, 0));
    
    cv::cvtColor(mask,mask,cv::COLOR_BGR2GRAY);
    cv::fillConvexPoly(mask,roi_points, 255);
    

    int threshold = 0, minLineLength = 0, maxLineGap = 0;
    cv::namedWindow("TrackBars",(640,400));
    
    cv::createTrackbar("Threshold","TrackBars",&threshold,200);
    cv::createTrackbar("MinLineLen","TrackBars",&minLineLength,200);
    cv::createTrackbar("MaxLineGap","TrackBars",&maxLineGap,20);

    for(int i=0;i<1108;i++){
        start = clock();

        img = cv::imread(path + std::to_string(i) + ".png",0);
        cv::Mat originalImg; 
        img.copyTo(originalImg);
        
        cv::bitwise_and(img, mask, img);
    
        cv::threshold(img, img, 130, 145, cv::THRESH_BINARY);
        
        std::vector<cv::Vec4i> linesP;
        
        cv::HoughLinesP(img, linesP, 1, CV_PI/180, threshold, minLineLength, maxLineGap); 
        
        // std::cout<<"Number of lines : "<<linesP.size()<<std::endl;
        for( size_t i = 0; i < linesP.size(); i++ )
        {
            cv::Vec4i l = linesP[i];
            cv::line( originalImg, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
        }

        end = clock();
        
        double seconds = (double(end) - double(start)) / double(CLOCKS_PER_SEC);
        double fpsLive = 1.0 / seconds;
        if(i % 20 == 0){
            std::cout<<"FPS : "<<fpsLive<<std::endl;
        }
        cv::putText(originalImg, "FPS : " + std::to_string(fpsLive), {20,20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1);

        cv::imshow("Frames", originalImg);

        if(cv::waitKey(1) >= 0) break;
    }
  
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}