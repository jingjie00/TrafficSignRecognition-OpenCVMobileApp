#include	"Supp.h"
#include	<iostream>
#include	<map>
#include	<math.h>
#include	<opencv2/core.hpp>
#include	<opencv2/dnn.hpp>
#include	<opencv2/features2d.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/imgproc.hpp>
#include	<opencv2/ml.hpp>
#include	"opencv2/opencv_modules.hpp"
#include	"opencv2/core/core.hpp"
#include	"opencv2/ml/ml.hpp"
#include	"opencv2/highgui/highgui.hpp"
#include	<opencv2/opencv.hpp>
#include	<opencv2/objdetect.hpp>
#include	<random>
#include	<vector>
#include    <fstream>

# define M_PI           3.14159265358979323846

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace dnn;

// Double Segmentation Function
std::array<Mat, 2> doubleSegment(Mat srcI2, Mat singleMask, int i) {

	const int		totalRow = srcI2.rows,
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

	std::array<Mat, 2> segmented;

	Mat temp;

	RNG				rng(12345);

	canvas.create(totalRow, totalCol, CV_8UC3);
	canvas = Scalar(0, 0, 0);
	canvas.copyTo(filledCanvasMask);

	std::vector<std::vector<Point> >	contours, contours_arrg;

	/*----------------------------------------------------------------------*/
	// Prepare display enviroment

	bilateralFilter(srcI2, blurring, size, size * 3, size / 3);

	// Convert the image to HSV Colour Space & Grayscale
	cvtColor(blurring, HSVImage, COLOR_BGR2HSV);

	// Declaring the Hue Ranges for the colours
	Scalar whiteLow = Scalar(0, 0, 0);
	Scalar whiteHigh = Scalar(180, 50, 255);
	Scalar blackLow = Scalar(0, 0, 0);
	Scalar blackHigh = Scalar(180, 50, 100);
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

	cvtColor(dilated, finalMask, COLOR_GRAY2BGR);
	//Segmented
	segmented[0] = finalMask & srcI2;

	segmented[0].copyTo(segmented[1]);
	segmented[0].setTo(Scalar(255, 255, 255), finalMask);

	return segmented;
}

void printImage300(std::vector<Mat> imageVector) {
	for (int i = 0; i < imageVector.size(); i++) {
		Mat temp;
		resize(imageVector[i], temp, Size(290, 290), INTER_LINEAR);
		imshow("Further Segment " + to_string(i + 1), temp);
		waitKey(0);
	}
}

// Get which is larger
bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i < j);
}

// Customised function
void fillHole(const Mat srcBw, Mat& dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());
	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	cv::floodFill(Temp, Point(0, 0), Scalar(255, 255, 255));
	Mat cutImg;
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	dstBw = srcBw | (~cutImg);
}

// Segmentation Function (Will Trigger Double Segmentation)
void segmentMain(String path, std::vector<Mat>& returnFurtherSegments, std::vector<std::string>& returnLabelNames, std::vector<Mat>& returnFirstSegments) {

	Mat		srcI = imread(path);
	/*imshow(path, srcI);
	waitKey(0);*/

	//resize(srcI,srcI, Size(320,480),0,0,1);

	static int		t1, t2, t3, t4;
	const int		totalRow = srcI.rows,
		totalCol = srcI.cols,
		ratio = 3,
		kernelSize = 3,
		size = 21;

	Mat		blurring,
		HSVImage,
		redMask1, redMask2, blueMask, yellowMask, mask,
		eroded,
		dilated,
		canvas,
		filledCanvasMask,
		segmented;

	Mat temp;

	RNG				rng(12345);

	canvas.create(totalRow, totalCol, CV_8UC3);
	canvas = Scalar(0, 0, 0);
	canvas.copyTo(filledCanvasMask);

	std::vector<std::vector<Point> >	contours, contours_arrg;

	bilateralFilter(srcI, blurring, size, size * 3, size / 3);

	// Convert the image to HSV Colour Space & Grayscale
	cvtColor(blurring, HSVImage, COLOR_BGR2HSV);
	cvtColor(HSVImage, temp, COLOR_HSV2BGR);

	// Declaring the Hue Ranges for the colours
	// Hue Ranges for Red
	// According to colour space, red is at the two left and right edge
	Scalar redLow1 = Scalar(159, 140, 160);
	Scalar redHigh1 = Scalar(180, 255, 255);

	Scalar redLow2 = Scalar(0, 50, 50);
	Scalar redHigh2 = Scalar(7, 255, 255);

	// Hue Ranges for Blue
	Scalar blueLow = Scalar(101, 150, 100);
	Scalar blueHigh = Scalar(130, 255, 255);

	// Hue Ranges for Yellow
	Scalar yellowLow = Scalar(16, 100, 140);
	Scalar yellowHigh = Scalar(36, 255, 255);

	// Match for Red
	inRange(HSVImage, redLow1, redHigh1, redMask1);
	inRange(HSVImage, redLow2, redHigh2, redMask2);

	// Match for Blue
	inRange(HSVImage, blueLow, blueHigh, blueMask);

	// Match for Yellow
	inRange(HSVImage, yellowLow, yellowHigh, yellowMask);

	// Remove noise
	Mat element(2, 2, CV_8U, cv::Scalar(1));
	morphologyEx(yellowMask, yellowMask, cv::MORPH_CLOSE, element);
	morphologyEx(redMask1, redMask1, cv::MORPH_CLOSE, element);
	morphologyEx(redMask2, redMask2, cv::MORPH_CLOSE, element);
	morphologyEx(blueMask, blueMask, cv::MORPH_CLOSE, element);

	// Merge the mask
	mask = redMask1 | redMask2 | blueMask | yellowMask;
	cvtColor(mask, temp, COLOR_GRAY2BGR);

	//erode
	erode(mask, eroded, element);
	cvtColor(eroded, temp, COLOR_GRAY2BGR);

	//dilate
	dilate(eroded, dilated, element);
	cvtColor(dilated, temp, COLOR_GRAY2BGR);

	// Find the contour
	findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

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

	// Sort according to size
	sort(contours.begin(), contours.end(), compareContourAreas);
	
	// draw the largest contour if exist
	if (contours.size() > 0) {
		contours_arrg.push_back(contours[contours.size() - 1]);
		drawContours(filledCanvasMask, contours_arrg, 0, Scalar(255, 255, 255));

		// Get the label names from the file
		std::string label_class = path.substr(15, 2);
		returnLabelNames.push_back(label_class);
	}

	fillHole(filledCanvasMask, filledCanvasMask);
	

	//Segmented
	segmented = filledCanvasMask & srcI;
	
	
	//Bounding box
	std::vector<std::vector<Point> > contours_poly(contours_arrg.size());
	std::vector<Rect> boundRect(contours_arrg.size());
	for (int i = 0; i < contours_arrg.size(); i++)
	{
		approxPolyDP(contours_arrg[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
	}

	Mat drawing = Mat::zeros(canvas.size(), CV_8UC3);
	for (int i = 0; i < contours_arrg.size(); i++)
	{
		Rect rect = boundRect[i];
		cv::rectangle(srcI, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 255, 0), 2, 0, 0);
	}

	// Crop and save
	for (int i = 0; i < boundRect.size(); i++) {
		// prepare only single mask
		Mat singleMask;
		singleMask.create(totalRow, totalCol, CV_8UC3);
		singleMask = Scalar(0, 0, 0);
		drawContours(singleMask, contours_arrg, i, Scalar(255, 255, 0));
		fillHole(singleMask, singleMask);

		// get the single traffic sign with rectangle
		Mat single = segmented & singleMask;
		returnFirstSegments.push_back(single);
		single(Rect(boundRect[i].tl().x, boundRect[i].tl().y, boundRect[i].br().x - boundRect[i].tl().x, boundRect[i].br().y - boundRect[i].tl().y)).copyTo(single);


		// Do double segment
		singleMask(Rect(boundRect[i].tl().x, boundRect[i].tl().y, boundRect[i].br().x - boundRect[i].tl().x, boundRect[i].br().y - boundRect[i].tl().y)).copyTo(singleMask);
		std::array<Mat, 2> seg = doubleSegment(single, singleMask, i);

		returnFurtherSegments.push_back(seg[0]);
	}
}

// Feature Extraction Function
void extractFeaturesFromTrafficSign(std::vector<Mat>& furtherSegments, std::vector<Mat>& initialSegments, std::vector<std::vector<double>>& retHuMomentfeature, std::vector<std::vector<float>>& retHogFeature) {

	// Calculate Hu Moments from Further Segments
	for (int i = 0; i < furtherSegments.size(); i++) {
		Mat temp;

		cvtColor(furtherSegments[i], temp, COLOR_BGR2GRAY);

		Moments moments = cv::moments(temp, true);
		std::vector<double> huMoments;
		std::vector<double> logged_huMoments;
		bool foundInf = false;
		HuMoments(moments, huMoments);

		// Push the log transform vectors
		for (int i = 0; i < 7; i++) {
			/*if (isinf(huMoments[i]) || huMoments[i] == 0) {
				foundInf = true;
			}*/
			// logged_huMoments.push_back(huMoments[i]);
			logged_huMoments.push_back(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])));
		}

		/*if (!foundInf)*/
		retHuMomentfeature.push_back(logged_huMoments);
	}

	HOGDescriptor hog;
	hog.winSize = Size(80, 80);
	hog.cellSize = Size(32, 32);
	hog.blockSize = Size(32, 32);
	hog.blockStride = Size(8, 8);
	vector<float> descriptors;

	for (size_t i = 0; i < initialSegments.size(); i++) {
		Mat temp;
		resize(initialSegments[i], temp, Size(80, 80), INTER_LINEAR);
		hog.compute(temp, descriptors);
		retHogFeature.push_back(descriptors);
	}

}

// Write to CSV
void generateFeatureMatrix(std::vector<std::vector<double>>& HuMomentfeature, std::vector<std::vector<float>>& HogFeature, std::vector<std::string>& labelVector) {

	// Label Encoding
	/*
	 017 : Horn -> 1
	 028 : Car -> 2
	 031 : Uturn -> 3
	 037 : School -> 4
	 054 : NoEntry -> 5
	 055 : Stop -> 6
	 */

	 // DEFINING A MAP TO STORE ALL LABEL CLASSES
	std::map<string, int> label_name = {
		{"17", 1},
		{"28", 2},
		{"31", 3},
		{"37", 4},
		{"54", 5},
		{"55", 6},
	};

	// Ofstream file to save csv
	std::ofstream featureCSV;
	featureCSV.open("traffic_sign_concat.csv");

	// Generate feature rows
	for (int i = 0; i < HuMomentfeature.size(); i++) {
		for (int j = 0; j < HuMomentfeature[i].size(); j++) {
			featureCSV << HuMomentfeature[i][j] << ",";
		}

		for (int j = 0; j < HogFeature[i].size(); j++) {
			featureCSV << HogFeature[i][j] << ",";
		}

		int val = 0;
		auto it = label_name.find(labelVector[i]);
		if (it != label_name.end())
			val = it->second;

		featureCSV << val;
		featureCSV << endl;
	}

	featureCSV.close();
}

int main(int argc, char** argv) {

	int choice;

	cout << "Skip Segmentation (1 - Yes, 2 - No): ";
	cin >> choice;

	if (choice == 2) {
		/*STEP 1 -> SEGMENTATION - WILL SAVE TO CSV*/

		/*PREPROCESSING - Variable Declaration*/
		std::vector<Mat> furtherSegments;
		std::vector<Mat> firstSegments;
		std::vector<std::string> allLabels;

		// Reading the input paths file
		ifstream inputPathsFile("Input Dataset/inputSignNames.txt");

		// Start Segmentation Process
		std::cout << "Begin Segmentation";

		int ct = 0;

		if (inputPathsFile.is_open()) {
			std::string name_temp;
			while (getline(inputPathsFile, name_temp)) {
				std::cout << ct++ << ". " << name_temp << " Done " << endl;
				segmentMain(name_temp, furtherSegments, allLabels, firstSegments); // returns -> 1) Internal Segment 2) Encoded label 3) Initial Segment
			}
		}

		std::cout << "\t....Done" << endl << endl;


		/*STEP 2 - Feature Extraction*/

		// Start Feature Extraction
		std::cout << "Begin Feature Extraction";

		// Feature Vector
		std::vector<std::vector<double>> huMomentsFeatures;
		std::vector<std::vector<float>> HoGFeatures;
		extractFeaturesFromTrafficSign(furtherSegments, firstSegments, huMomentsFeatures, HoGFeatures);

		std::cout << "\t....Done" << endl << endl;


		/*STEP 3 - Writing Feature to CSV*/

		// Start Write To CSV
		std::cout << "Begin writing to CSV" ;

		generateFeatureMatrix(huMomentsFeatures, HoGFeatures, allLabels);

		std::cout << "\t....Done" << endl << endl;
	}
	
	/*STEP 4 - DATASET PREPROCESSING - Reading the csv dataset file*/

	Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::loadFromCSV("traffic_sign_concat_imputed.csv", 0);
	if (dataset.empty()) {
		std::cout << "CSV can't be read" << endl;
		return 0;
	}
	
	// Splitting the dataset into training and testing (70% Training and 30% Testing)
	dataset->setTrainTestSplitRatio(0.70);
	std::cout << "No. of Test / No. of Train: " << dataset->getNTestSamples() << " / " << dataset->getNTrainSamples() << endl;
	
	Mat trainSet = dataset->getTrainSamples();
	Mat trainLabels = dataset->getTrainResponses();

	Mat testSet = dataset->getTestSamples();
	Mat testLabels = dataset->getTestResponses();

	// Converting label to use as integer for SVM
	Mat trainLabels_int;
	trainLabels.convertTo(trainLabels_int, CV_32S);

	Mat testLabels_int;
	testLabels.convertTo(testLabels_int, CV_32S);
	std::system("Pause");
	std::system("cls");

	// Print out the training data
	std::cout << "=====================================" << endl;
	std::cout << "Training Data Matrix" << endl;
	std::cout << "=====================================" << endl << endl;
	std::cout << "\t\t\t" << "Hu Moment 1\t" << "Hu Moment 2\t" << "Hu Moment 3\t" << "Hu Moment 4\t" << "Hu Moment 5\t" << "Hu Moment 6\t" << "Hu Moment 7\t" << endl << endl;
	for (int i = 0; i < trainSet.rows; i++) {
		std::cout << "Training Sample " << i + 1 << "\t";
		for (int j = 0; j < 7; j++) {
			std::cout << setw(10) << trainSet.at<float>(i, j) << "\t";
		}
		std::cout << endl;
	}
	std::cout << endl << endl;

	// Print out the test data
	std::cout << "=====================================" << endl;
	std::cout << "Test Data Matrix" << endl;
	std::cout << "=====================================" << endl << endl;
	std::cout << "\t\t\t" << "Hu Moment 1\t" << "Hu Moment 2\t" << "Hu Moment 3\t" << "Hu Moment 4\t" << "Hu Moment 5\t" << "Hu Moment 6\t" << "Hu Moment 7\t" << endl << endl;
	for (int i = 0; i < testSet.rows; i++) {
		std::cout << "Testing Sample " << i + 1 << "\t";
		for (int j = 0; j < 7; j++) {
			std::cout << setw(10) << testSet.at<float>(i, j) << "\t";
		}
		std::cout << endl;
	}
	std::cout << endl << endl;

	std::cout << "Begin Model Training - SVM + RF" << endl << endl;

	/*STEP 6- MODEL TRAINING*/

	/*Random Forest*/
	Ptr<RTrees> rtrees = RTrees::create();
	rtrees->setMaxDepth(10);
	rtrees->setMinSampleCount(2);
	rtrees->setRegressionAccuracy(0);
	rtrees->setUseSurrogates(false);
	rtrees->setMaxCategories(16);
	rtrees->setPriors(Mat());
	rtrees->setCalculateVarImportance(true);
	rtrees->setActiveVarCount(0);
	rtrees->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0));

	// Training
	std::cout << "Begin Training Random Forest";
	rtrees->train(trainSet, ROW_SAMPLE, trainLabels_int);
	rtrees->save("T6G1_rtree.xml");
	std::cout << "\t....Done" << endl << endl;
	// Testing 
	std::cout << "Begin prediction of testing set";
	Mat rtrees_testResults;
	rtrees->predict(testSet, rtrees_testResults);
	std::cout << "\t....Done" << endl << endl;

	/*SVM*/
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));


	// Training
	std::cout << "Begin Training SVM";
	svm->train(trainSet, ROW_SAMPLE, trainLabels_int);
	svm->save("T6G1_SVM.xml");
	std::cout << "\t....Done" << endl << endl;
	// Testing
	std::cout << "Begin prediction of testing set";
	Mat svm_testResults;
	svm->predict(testSet, svm_testResults);
	std::cout << "\t....Done" << endl << endl;
	std::cout << "Done Model Training - SVM + RF" << endl << endl;
	
	std::cout << "Experimental Results - Accuracy of Classifier" << endl << endl;

	/*STEP 7 - EXPERIMENTAL RESULTS*/
	ofstream outRTree("Random_Forest_res.csv");
	ofstream outSVM("SVM_res.csv");
	ofstream outTestLabels("testLabels.csv");

	// Random Forest Results
	float rtree_accuracy = float(countNonZero(rtrees_testResults == testLabels)) / testLabels.rows;
	std::cout << "\n\nRtree Classification accuracy: " << rtree_accuracy;
	std::cout << endl;

	// SVM Results
	float svm_accuracy = float(countNonZero(svm_testResults == testLabels)) / testLabels.rows;
	std::cout << "\n\nSVM Classification accuracy: " << svm_accuracy;
	std::cout << endl;

	// Write the Predicted Labels to CSV (Use Python for Analysis)
	outRTree << format(rtrees_testResults, cv::Formatter::FMT_CSV) << endl;
	outSVM << format(svm_testResults, cv::Formatter::FMT_CSV) << endl;
	outTestLabels << format(testLabels, cv::Formatter::FMT_CSV) << endl;

	outRTree.close();
	outSVM.close();
	outTestLabels.close();
	
	std::cout << endl << endl;
	std::system("pause");
	return 0;
}
