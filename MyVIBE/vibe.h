#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define NUM_SAMPLES 20 //每个像素点的样本个数
#define MIN_MATCHES 2
#define RADIUS 12
#define SUBSAMPLE_FACTOR 1000
#define SCALEADD 40
class ViBe_BGS
{
public:
	void init(const Mat _image); //初始化  
	void processFirstFrame(const Mat _image);
	void testAndUpdate(const Mat _image); //更新 
	vector<Rect> ROIget(ViBe_BGS &Vibe_Bgs, Mat &frame, int frameCount);
	Mat getMask(void){ return m_mask; };
	int scaleAdd(Rect &rectangle, int width, int height);
	bool isOverlap(Rect r1, Rect r2);
	Rect mergeRect(Rect r1, Rect r2);
	vector<Rect> mergeFrameRect(vector<Rect> rect);
private:
	Mat m_samples[NUM_SAMPLES];
	Mat m_foregroundMatchCount;
	Mat m_mask;
};