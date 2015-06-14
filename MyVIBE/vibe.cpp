#include <opencv2/opencv.hpp>
#include <iostream>
#include "ViBe.h"

using namespace std;
using namespace cv;

int c_xoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };  //x的邻居点
int c_yoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };  //y的邻居点



/**************** Assign space and init ***************************/
void ViBe_BGS::init(const Mat _image)
{
	for (int i = 0; i < NUM_SAMPLES; i++)
	{
		m_samples[i] = Mat::zeros(_image.size(), CV_8UC1);
	}
	m_mask = Mat::zeros(_image.size(), CV_8UC1);
	m_foregroundMatchCount = Mat::zeros(_image.size(), CV_8UC1);
}

/**************** Init model from first frame ********************/
void ViBe_BGS::processFirstFrame(const Mat _image)
{
	RNG rng;
	int row, col;

	for (int i = 0; i < _image.rows; i++)
	{
		for (int j = 0; j < _image.cols; j++)
		{
			for (int k = 0; k < NUM_SAMPLES; k++)
			{
				// Random pick up NUM_SAMPLES pixel in neighbourhood to construct the model
				int random = rng.uniform(0, 9);

				row = i + c_yoff[random];
				if (row < 0)
					row = 0;
				if (row >= _image.rows)
					row = _image.rows - 1;

				col = j + c_xoff[random];
				if (col < 0)
					col = 0;
				if (col >= _image.cols)
					col = _image.cols - 1;

				m_samples[k].at<uchar>(i, j) = _image.at<uchar>(row, col);
			}
		}
	}
}

/**************** Test a new frame and update model ********************/
void ViBe_BGS::testAndUpdate(const Mat _image)
{
	RNG rng;

	for (int i = 0; i < _image.rows; i++){
		for (int j = 0; j < _image.cols; j++){
			int matches(0), count(0);
			float dist;
			while (matches < MIN_MATCHES && count < NUM_SAMPLES){
				dist = abs(m_samples[count].at<uchar>(i, j) - _image.at<uchar>(i, j));
				if (dist < RADIUS)
					matches++;
				count++;
			}

			if (matches >= MIN_MATCHES){
				// It is a background pixel
				m_foregroundMatchCount.at<uchar>(i, j) = 0;
				// Set background pixel to 0
				m_mask.at<uchar>(i, j) = 0;
				// 如果一个像素是背景点，那么它有 1 / defaultSubsamplingFactor 的概率去更新自己的模型样本值
				int random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0){
					random = rng.uniform(0, NUM_SAMPLES);
					m_samples[random].at<uchar>(i, j) = _image.at<uchar>(i, j);
				}
				// 同时也有 1 / defaultSubsamplingFactor 的概率去更新它的邻居点的模型样本值
				random = rng.uniform(0, SUBSAMPLE_FACTOR);
				if (random == 0){
					int row, col;
					random = rng.uniform(0, 9);
					row = i + c_yoff[random];
					if (row < 0)
						row = 0;
					if (row >= _image.rows)
						row = _image.rows - 1;

					random = rng.uniform(0, 9);
					col = j + c_xoff[random];
					if (col < 0)
						col = 0;
					if (col >= _image.cols)
						col = _image.cols - 1;

					random = rng.uniform(0, NUM_SAMPLES);
					m_samples[random].at<uchar>(row, col) = _image.at<uchar>(i, j);
				}
			}
			else{
				// It is a foreground pixel
				m_foregroundMatchCount.at<uchar>(i, j)++;
				// Set background pixel to 255
				m_mask.at<uchar>(i, j) = 255;
				//如果某个像素点连续N次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
				if (m_foregroundMatchCount.at<uchar>(i, j) > 50){
					int random = rng.uniform(0, SUBSAMPLE_FACTOR);
					if (random == 0){
						random = rng.uniform(0, NUM_SAMPLES);
						m_samples[random].at<uchar>(i, j) = _image.at<uchar>(i, j);
					}
				}
			}
		}
	}
}

//获取运动目标的矩形区域
vector<Rect> ViBe_BGS::ROIget(ViBe_BGS &Vibe_Bgs, Mat &frame, int frameCount)
{
	Mat img;
	frame.copyTo(img);
	int width, height;
	height = frame.rows;
	width = frame.cols;
	if (!frame.data){
		cerr << "读取图片失败！" << endl;
		exit(0);
	}
	Mat gray, mask;
	vector<Rect> maskRect;
	cvtColor(frame, gray, CV_RGB2GRAY);
	//初始化VIBE   model
	if (frameCount == 1) {
		Vibe_Bgs.init(gray);
		Vibe_Bgs.processFirstFrame(gray);
		cout << " init VIBE complete!" << endl;
	}
	else{
		Vibe_Bgs.testAndUpdate(gray);
		mask = Vibe_Bgs.getMask();
		erode(mask, mask, Mat(5, 5, CV_8U));
		dilate(mask, mask, Mat(10, 10, CV_8U));
		vector<vector<Point>> contours;
		imshow("Mask",mask);
		waitKey(0);
		findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		Rect maskRectTemp;
		for (vector<vector<Point>>::iterator it = contours.begin(); it != contours.end();){
			if (it->size() < 100){
				it = contours.erase(it);
			}
			else{
				maskRectTemp = boundingRect(*it);
				scaleAdd(maskRectTemp, width, height);
				maskRect.push_back(maskRectTemp);
				rectangle(img, maskRectTemp, Scalar(0, 0, 255), 2);
				++it;
			}
		}
	}

	//imshow("input", frame);
	//waitKey(10);
	//cout << frameCount << "-" << maskRect.size() << ends;
	imshow("img", img);
	waitKey(10);
	maskRect = mergeFrameRect(maskRect);
	return maskRect;
}

int ViBe_BGS::scaleAdd(Rect &rectangle, int width, int height)
{
	rectangle.x = rectangle.x - SCALEADD < 0 ? 0 : rectangle.x - SCALEADD;
	rectangle.y = rectangle.y - SCALEADD < 0 ? 0 : rectangle.y - SCALEADD;
	rectangle.width = rectangle.x + rectangle.width + 2 * SCALEADD >= width ?
		width - rectangle.x : rectangle.width + 2 * SCALEADD;
	rectangle.height = rectangle.y + rectangle.height + 2 * SCALEADD >= height ?
		height - rectangle.y : rectangle.height + 2 * SCALEADD;
	return 0;
}
//合并图片中的若干矩形
vector<Rect> ViBe_BGS::mergeFrameRect(vector<Rect> rect)
{
	uchar rectMergeFlage[1000];
	memset(rectMergeFlage, 0, sizeof(rectMergeFlage));
	int count = rect.size();
	for (size_t i = 0; i < count; i++){
		if (rectMergeFlage[i] == 0){
			for (size_t j = 0; j < count; j++){
				if (rectMergeFlage[i] == 0 && rectMergeFlage[j] == 0 && j != i && isOverlap(rect[i], rect[j])){
					rect[i] = mergeRect(rect[i], rect[j]);
					rectMergeFlage[j] = 1;
					j = 0;
				}
			}
		}
	}
	vector<Rect> tempRect;
	for (size_t i = 0; i < count; i++){
		if (rectMergeFlage[i] == 0){
			tempRect.push_back(rect[i]);
		}
	}
	return tempRect;
}

//判断连个矩形是否相交或包含
bool ViBe_BGS::isOverlap(Rect r1, Rect r2)
{
	int minx, miny;
	int maxx, maxy;
	minx = max(r1.x, r2.x);
	maxx = min(r1.x + r1.width, r2.x + r2.width);
	miny = max(r1.y, r2.y);
	maxy = min(r1.y + r1.height, r2.y + r2.height);
	if (minx <= maxx && miny <= maxy){
		return true;
	}
	if (r1.x <= r2.x && r1.x + r1.width >= r2.x + r2.width && r1.y <= r2.y && r1.y + r1.height >= r2.y + r2.height){
		return true;
	}
	if (r1.x >= r2.x && r1.x + r1.width <= r2.x + r2.width && r1.y >= r2.y && r1.y + r1.height <= r2.y + r2.height){
		return true;
	}
	return false;
}
//合并两个矩形
Rect ViBe_BGS::mergeRect(Rect r1, Rect r2)
{
	Rect rect;
	rect.x = min(r1.x, r2.x);
	rect.y = min(r1.y, r2.y);
	rect.width = max(r1.x + r1.width - rect.x, r2.x + r2.width - rect.x);
	rect.height = max(r1.y + r1.height - rect.y, r2.y + r2.height - rect.y);
	return rect;
}
