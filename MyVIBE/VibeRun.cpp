#include <opencv2/imgproc/imgproc.hpp>
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
	vector<Rect> rectset;
	if (v.isOpened()){
		Mat frame;
		long long count = 0;
		while (v.read(frame))
		{
			++count;
			rectset = vibeModel.ROIget(vibeModel, frame, count);
			cout << count << endl;
			for (size_t i = 0; i < rectset.size(); i++){
				rectangle(frame,rectset[i],Scalar(0,0,255),1);
				cout << "x1: " << rectset[i].x << " y1: " << rectset[i].y << "   x2: " << rectset[i].x + rectset[i].width << " y2: " << rectset[i].y + rectset[i].height << endl;
			}
			imshow("img",frame);
			waitKey(1000);
		}
	}
	return 0;
}