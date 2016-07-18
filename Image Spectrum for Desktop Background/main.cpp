#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <dirent.h>
#include <vector>

using namespace std;
using namespace cv;

struct HSV{
	HSV(float h, float s, float v) {
		this->h = h;
		this->s = s;
		this->v = v;
	}
	float h;
	float s;
	float v;
};

struct ImageWithHue {
	ImageWithHue(Mat image, float h) {
		this->image = image;
		this->h = h;
	}
	Mat image;
	float h;
};

struct LessThanHue {
	inline bool operator() (const ImageWithHue& o1, const ImageWithHue& o2) {
		return (o1.h < o2.h);
	}
};

HSV findDominantColor(Mat image);
float findMax(Mat ch);
void RGBtoHSV(float r, float g, float b, float *h, float *s, float *v);

int main() {
	DIR * dir;
	struct dirent *ent;
	vector<ImageWithHue> images;
	if ((dir = opendir(".\\images\\")) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
			string fileName = ent->d_name;
			// skip these
			if (fileName == "." || fileName == "..")
			{
				continue;
			}
			Mat image = imread("./images/" + fileName);
			if (image.empty())
			{
				cout << "Image not read: " << fileName << endl;
				system("pause");
				return -1;
			}
			cout << fileName << ":" << endl;
			//namedWindow("Original", WINDOW_AUTOSIZE);
			//imshow("Original", image);
			HSV hsv = findDominantColor(image);
			cout << "H: " << hsv.h << endl;
			cout << "S: " << hsv.s << endl;
			cout << "V: " << hsv.v << endl;
			images.push_back(ImageWithHue(image, hsv.h));
		}
		std::sort(images.begin(), images.end(), LessThanHue());

		for (int i = 0; i < images.size(); i++)
		{
			string imageName = "./images-arranged/" + std::to_string(i) + ".jpg";
			imwrite(imageName, images[i].image);
		}
	}else{
		return -1;
	}
	waitKey(0);
	return 0;
}

/* Return the hue of the image */
HSV findDominantColor(Mat inputImage) {

	Mat image;
	resize(inputImage, image, Size(inputImage.rows / 2, inputImage.cols / 2));

	std::vector<Mat> channels;
	split(image, channels);

	Mat bCh = channels[0];
	Mat gCh = channels[1];
	Mat rCh = channels[2];

	vector<float> bArray;
	vector<float> gArray;
	vector<float> rArray;
	if (bCh.isContinuous()) {
		bArray.assign(bCh.datastart, bCh.dataend);
	}
	if (gCh.isContinuous()) {
		gArray.assign(gCh.datastart, gCh.dataend);
	}
	if (rCh.isContinuous()) {
		rArray.assign(rCh.datastart, rCh.dataend);
	}
	Mat bMat = Mat(bArray, CV_32F);
	Mat gMat = Mat(gArray, CV_32F);
	Mat rMat = Mat(rArray, CV_32F);

	//cout << b_hist << endl;
	Mat labels;
	Mat bCenters, gCenters, rCenters;
	cout << "k-means blue..." << endl;
	kmeans(bMat, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, bCenters);
	cout << "k-means green..." << endl;
	kmeans(gMat, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, gCenters);
	cout << "k-means red..." << endl;
	kmeans(rMat, 3, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, rCenters);

	
	float b = findMax(bCenters) / 255;
	float g = findMax(gCenters) / 255;
	float r = findMax(rCenters) / 255;

	//cout << "R: " << r * 255 << "  " << r << endl;
	//cout << "G: " << g * 255 << "  " << g << endl;
	//cout << "B: " << b * 255 << "  " << b << endl;

	float h, s, v;
	RGBtoHSV(r, g, b, &h, &s, &v);
	//cout << "Hue: " << h << endl;
	return HSV(h, s, v);
}

float findMax(Mat ch) {
	float max = 0;
	for (int i = 0; i < ch.rows; i++)
	{
		float cur = ch.at<float>(i, 0);
		if (max < cur)
		{
			max = cur;
		}
	}
	return  max;
}

void RGBtoHSV(float r, float g, float b, float *h, float *s, float *v)
{
	float mi, max, delta;
	mi = std::min(std::min(r, g), b);
	max = std::max(std::max(r, g), b);
	*v = max;
	delta = max - mi;
	if (max != 0)
		*s = delta / max;		// s
	else {
		// r = g = b = 0		// s = 0, v is undefined
		*s = 0;
		*h = -1;
		return;
	}
	if (r == max)
		*h = (g - b) / delta;		// between yellow & magenta
	else if (g == max)
		*h = 2 + (b - r) / delta;	// between cyan & yellow
	else
		*h = 4 + (r - g) / delta;	// between magenta & cyan
	*h *= 60;				// degrees
	if (*h < 0)
		*h += 360;
}