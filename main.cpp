/*************************************************

Copyright: Guangyu Zhong all rights reserved

Author: Guangyu Zhong

Date:2014-09-27

Description: codes for Manifold Ranking Saliency Detection
Reference http://ice.dlut.edu.cn/lu/Project/CVPR13[yangchuan]/cvprsaliency.htm

**************************************************/
#include<iostream>
#include<cv.h>
#include<highgui.h>
#include"PreGraph.h"
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	
	string filename = argv[1];
	float thr = 100;
	if (argc > 2)
		thr = atof(argv[2]);
	Mat img = imread(filename);

	PreGraph SpMat;
	Mat superpixels = SpMat.GeneSp(img);
	Mat sal = SpMat.GeneSal(img);
	//cout << sal;
	Mat salMap = SpMat.Sal2Img(superpixels, sal);
	/*
	Mat tmpsuperpixels;
	normalize(salMap, tmpsuperpixels, 255.0, 0.0, NORM_MINMAX);
	tmpsuperpixels.convertTo(tmpsuperpixels, CV_8UC3, 1.0);
	imshow("sp", tmpsuperpixels);
	waitKey();
	*/
	Mat mask;
	normalize(salMap, mask, 255.0, 0.0, NORM_MINMAX);
	mask.convertTo(mask, CV_8UC3, 1.0);
	mask = mask > thr;
	Mat img1 = cv::Mat::zeros(img.size(), CV_8UC3);
	img.copyTo(img1, mask);
	imshow("sp", img1);

	return 0;
}
