#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "vibe.h"
using namespace std;
using namespace std;
char path[256] = "d:/Data/test/1.ts";
int main()
{

	ViBe_BGS vibeModel;
	VideoCapture v;
	v.open("d:/Data/test/1.ts");
	Mat frame;
	long long count = 0;
	v >> frame;
	while (frame.data)
	{
		++count;
		vibeModel.ROIget(vibeModel, frame, count);
		//imshow("img",frame);
		//waitKey(10);
		v >> frame;
	}
	system("pause");
	return 0;
}