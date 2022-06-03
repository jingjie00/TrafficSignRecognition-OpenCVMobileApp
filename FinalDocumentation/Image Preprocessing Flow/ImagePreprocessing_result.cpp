#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/imgproc.hpp>
#include	<opencv2/core.hpp>
#include	<iostream>
#include	<direct.h>
#include    <fstream>
#include	<math.h>
#include	"Supp.h"
#include <filesystem>
namespace fs = std::filesystem;

# define M_PI           3.14159265358979323846
using namespace cv;
using namespace std;


// Get which is larger
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}

// Customised function
void fillHole(const Mat srcBw, Mat& dstBw)
{
	//imshow("Original", srcBw);
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	//imshow("Temp", Temp);
	cv::floodFill(Temp, Point(0, 0), Scalar(255, 255, 255));
	//imshow("After floodFill() Temp", Temp);
	Mat cutImg;
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	//imshow("CutImg", cutImg);
	//imshow("~CutImg", ~cutImg);
	dstBw = srcBw | (~cutImg);
	//imshow("Final Image", dstBw);
}

array<Mat, 2> furtherSegment(Mat srcI2, Mat singleMask, String name, int i) {
	Mat		largeWin, win[1 * 3], // create the new window
		legend[1 * 3]; // and the means to each sub-window
	createWindowPartition(srcI2, largeWin, win, legend, 1, 3);
	srcI2.copyTo(win[0]);
	putText(legend[0], "First Segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	const int		noOfImagePerCol = 2, noOfImagePerRow = 5, // create a 3X4 window partition
		totalRow = srcI2.rows,
		totalCol = srcI2.cols,
		ratio = 3,
		kernelSize = 3,
		size = 21;

	Mat	blurring,
		HSVImage,
		white, black, blue, mask,
		eroded,
		dilated,
		canvas,
		filledCanvasMask,
		finalMask;

	array<Mat, 2> segmented;

	Mat temp;

	RNG				rng(12345);

	canvas.create(totalRow, totalCol, CV_8UC3);
	canvas = Scalar(0, 0, 0);
	canvas.copyTo(filledCanvasMask);

	vector<vector<Point> >	contours, contours_arrg;

	/* ---------------------------------------------------------------------- */
	// Prepare display enviroment
	bilateralFilter(srcI2, blurring, size, size * 3, size / 3);

	// Convert the image to HSV Colour Space & Grayscale
	cvtColor(blurring, HSVImage, COLOR_BGR2HSV);

	// Declaring the Hue Ranges for the colours
	Scalar whiteLow = Scalar(0, 0, 150);
	Scalar whiteHigh = Scalar(180, 50, 255);
	Scalar blackLow = Scalar(0, 0, 0);
	Scalar blackHigh = Scalar(180, 100, 60);
	Scalar blueLow = Scalar(104, 236, 0);
	Scalar blueHigh = Scalar(110, 255, 255);

	// Match for Red
	inRange(HSVImage, whiteLow, whiteHigh, white);
	inRange(HSVImage, blackLow, blackHigh, black);
	inRange(HSVImage, blueLow, blueHigh, blue);

	// Remove noice
	Mat element(2, 2, CV_8U, cv::Scalar(1));
	morphologyEx(white, white, cv::MORPH_CLOSE, element);
	morphologyEx(black, black, cv::MORPH_CLOSE, element);
	morphologyEx(blue, blue, cv::MORPH_CLOSE, element);

	// Merge the mask
	cvtColor(singleMask, singleMask, COLOR_BGR2GRAY);
	mask = (white | black | blue) & singleMask;

	erode(mask, eroded, element);
	dilate(eroded, dilated, element);

	//Eliminate unneccesary contours
	//findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//drawContours(canvas, contours, contours.size()-1, Scalar(255, 255, 255)); // draw boundariess
	//sort(contours.begin(), contours.end(), compareContourAreas);
	//fillHole(canvas, filledCanvasMask);

	cvtColor(dilated, finalMask, COLOR_GRAY2BGR);
	//Segmented
	segmented[0] = finalMask & srcI2;
	segmented[0].copyTo(win[1]);
	putText(legend[1], "Further Segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	segmented[0].copyTo(segmented[1]);
	segmented[1].setTo(Scalar(255, 255, 255), finalMask);
	segmented[1].copyTo(win[2]);
	resize(segmented[1], segmented[1], Size(80, 80), INTER_LINEAR);
	putText(legend[2], "Binary output", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	//Show content
	imshow("Further Segmentation", largeWin);

	// Record
	imwrite("Output/" + name + "/Result/FurtherSegmentation" + to_string(i) + ".png", largeWin);

	return segmented;
}

void segmentMain(Mat in, String name) {

	Mat		srcI  = in;

	//resize(srcI,srcI, Size(320,480),0,0,1);

	static int		t1, t2, t3, t4;
	const int		noOfImagePerCol = 2, noOfImagePerRow = 5, // create a 3X4 window partition
		totalRow = srcI.rows,
		totalCol = srcI.cols,
		ratio = 3,
		kernelSize = 3,
		size = 21;

	Mat		largeWin, summaryLargeWin, summaryWin[1 * 2], win[noOfImagePerRow * noOfImagePerCol], // create the new window
		summaryLegend[1 * 2], legend[noOfImagePerRow * noOfImagePerCol]; // and the means to each sub-window

	Mat		blurring,
		HSVImage,
		redMask1, redMask2, blueMask, yellowMask, mask,
		eroded,
		dilated,
		canvas,
		finalCanvas,
		finalCanvasMask,
		segmented;

	Mat temp;

	RNG				rng(12345);

	canvas.create(totalRow, totalCol, CV_8UC3);
	canvas = Scalar(0, 0, 0);
	canvas.copyTo(finalCanvasMask);

	vector<vector<Point> >	contours, contours_arrg;

	/* ---------------------------------------------------------------------- */
		// Prepare display enviroment
	createWindowPartition(srcI, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);
	createWindowPartition(srcI, summaryLargeWin, summaryWin, summaryLegend, 1, 2);


	putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	srcI.copyTo(win[0]);
	putText(summaryLegend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	srcI.copyTo(summaryWin[0]);

	bilateralFilter(srcI, blurring, size, size * 3, size / 3);
	putText(legend[1], "Bilateral Filter", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	blurring.copyTo(win[1]);

	// Convert the image to HSV Colour Space & Grayscale
	cvtColor(blurring, HSVImage, COLOR_BGR2HSV);
	cvtColor(HSVImage, temp, COLOR_HSV2BGR);
	putText(legend[2], "HSV converted", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	temp.copyTo(win[2]);

	// Declaring the Hue Ranges for the colours
	// Hue Ranges for Red
	// According to color space, red is at the two left and right edge
	Scalar redLow1 = Scalar(150, 140, 160);
	Scalar redHigh1 = Scalar(180, 255, 255);

	Scalar redLow2 = Scalar(0, 50, 50);
	Scalar redHigh2 = Scalar(3, 255, 255);

	// Hue Ranges for Blue
	Scalar blueLow = Scalar(100, 150, 100);
	Scalar blueHigh = Scalar(128, 255, 255);

	// Hue Ranges for Yellow
	Scalar yellowLow = Scalar(14, 100, 140);
	Scalar yellowHigh = Scalar(30, 255, 255);

	// Match for Red
	inRange(HSVImage, redLow1, redHigh1, redMask1);
	inRange(HSVImage, redLow2, redHigh2, redMask2);

	// Match for Blue
	inRange(HSVImage, blueLow, blueHigh, blueMask);

	// Match for Yellow
	inRange(HSVImage, yellowLow, yellowHigh, yellowMask);



	// Remove noice
	Mat element(2, 2, CV_8U, cv::Scalar(1));
	morphologyEx(yellowMask, yellowMask, cv::MORPH_CLOSE, element);
	morphologyEx(redMask1, redMask1, cv::MORPH_CLOSE, element);
	morphologyEx(redMask2, redMask2, cv::MORPH_CLOSE, element);
	morphologyEx(blueMask, blueMask, cv::MORPH_CLOSE, element);

	//imshow("Red mask1", redMask1);
	//imshow("Red mask2", redMask2);
	//imshow("Red concate", redMask1 | redMask2);

	// Merge the mask
	mask = redMask1 | redMask2 | blueMask | yellowMask;
	cvtColor(mask, temp, COLOR_GRAY2BGR);
	temp.copyTo(win[3]);
	putText(legend[3], "Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	//erode
	erode(mask, eroded, element);
	cvtColor(eroded, temp, COLOR_GRAY2BGR);
	temp.copyTo(win[4]);
	putText(legend[4], "Eroded", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	//dilate
	dilate(eroded, dilated, element);
	cvtColor(dilated, temp, COLOR_GRAY2BGR);
	temp.copyTo(win[5]);
	putText(legend[5], "Dilate", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	// Find the contour
	findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//Display purpose(called canvas)
	for (int i = 0; i < contours.size(); i++) { // Just in case there is more than one object in image
		for (;;) { // get random colors that are not too dim
			t1 = rng.uniform(0, 255); // blue
			t2 = rng.uniform(0, 255); // green
			t3 = rng.uniform(0, 255); // red
			t4 = t1 + t2 + t3;
			if (t4 > 255) break;
		}

		drawContours(canvas, contours, i, Scalar(t1, t2, t3)); // draw boundariess

	}
	putText(legend[6], "Canvas", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	canvas.copyTo(win[6]);

	// Sort according to size
	sort(contours.begin(), contours.end(), compareContourAreas);

	// Eliminate smaller traffic sign
	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) * 6 > contourArea(contours[contours.size() - 1]))
			contours_arrg.push_back(contours[i]);
	}


	// Do the final canvas (flood)
	for (int i = 0; i < contours_arrg.size(); i++) { // Just in case there is more than one object in image

		drawContours(finalCanvasMask, contours_arrg, i, Scalar(255, 255, 255)); // draw boundariess
	}

	finalCanvasMask.copyTo(win[7]);
	putText(legend[7], "Final canvas boundary", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

	fillHole(finalCanvasMask, finalCanvasMask);
	putText(legend[8], "Final mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	finalCanvasMask.copyTo(win[8]);

	//Segmented
	segmented = finalCanvasMask & srcI;
	putText(legend[9], "Final segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	segmented.copyTo(win[9]);
	putText(summaryLegend[1], "Final segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
	segmented.copyTo(summaryWin[1]);

	//Bounding box
	vector<vector<Point> > contours_poly(contours_arrg.size());
	vector<Rect> boundRect(contours_arrg.size());
	for (int i = 0; i < contours_arrg.size(); i++)
	{
		approxPolyDP(contours_arrg[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
	}


	// Crop and save
	for (int i = 0; i < boundRect.size(); i++) {

		//prepare only single mask
		Mat singleMask, resized;
		singleMask.create(totalRow, totalCol, CV_8UC3);
		singleMask = Scalar(0, 0, 0);
		drawContours(singleMask, contours_arrg, i, Scalar(255, 255, 0));
		fillHole(singleMask, singleMask);

		//get the single traffic sign with rectangle
		Mat single = segmented & singleMask;
		single(Rect(boundRect[i].tl().x, boundRect[i].tl().y, boundRect[i].br().x - boundRect[i].tl().x, boundRect[i].br().y - boundRect[i].tl().y)).copyTo(single);
		resize(single, resized, Size(80, 80), INTER_LINEAR);
		imwrite("Output/" + name + "/Segmented/Img_" + to_string(i) + ".png", resized);


		//Do double segment
		singleMask(Rect(boundRect[i].tl().x, boundRect[i].tl().y, boundRect[i].br().x - boundRect[i].tl().x, boundRect[i].br().y - boundRect[i].tl().y)).copyTo(singleMask);
		array<Mat, 2> seg = furtherSegment(single, singleMask, name,i);
		imwrite("Output/" + name + "/FurtherSegmented/Img_" + to_string(i) + ".png", seg[0]);
		imwrite("Output/" + name + "/BinarySegment/Img_" + to_string(i) + ".png", seg[1]);
	}

	//Show content
	imshow("Image processing", largeWin);
	imshow("Summary Image processing", summaryLargeWin);

	// Record
	imwrite("Output/" + name + "/Result/Flow.png", largeWin);
}

int main(int argc, char** argv) {
	fs::remove_all("Output");
	ifstream inputPathsFile("Inputs/inputSignNames.txt");
	int count = 1;
	if (inputPathsFile.is_open()) {
		string name_temp;
		while (getline(inputPathsFile, name_temp)) {
			

			fs::create_directories("Output/Img_" + to_string(count) + "/Result");

			fs::create_directories("Output/Img_" + to_string(count) + "/Segmented");

			fs::create_directories("Output/Img_" + to_string(count) + "/FurtherSegmented");

			fs::create_directories("Output/Img_" + to_string(count) + "/BinarySegment");

			Mat in = imread(name_temp);
			segmentMain(in, "Img_"+ to_string(count));
			count++;
		}
	}



	waitKey();
	return 0;
}