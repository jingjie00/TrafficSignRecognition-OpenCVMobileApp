#include	<iostream>
#include	<opencv2/core/core.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<opencv2/features2d.hpp>
#include	<opencv2/imgproc.hpp>
#include	<opencv2/objdetect.hpp>
#include	<opencv2/ml.hpp>
#include	<random>
#include	<vector>
#include    <fstream>
#include	<map>
#include	<iomanip>

using namespace std;
using namespace cv;

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
	Scalar whiteLow = Scalar(0, 0, 150);
	Scalar whiteHigh = Scalar(180, 50, 255);
	Scalar blackLow = Scalar(0, 0, 0);
	Scalar blackHigh = Scalar(180, 50, 60);
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

// Print Image with size 300 by 300
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
void segmentMain(String path, std::vector<Mat>& returnFurtherSegments, std::vector<std::string>& returnLabelNames, std::vector<Mat>& returnFirstSegments, std::vector<std::string>& returnPaths) {

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
		returnPaths.push_back(path);
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
void generateFeatureMatrix(std::vector<std::vector<double>>& HuMomentfeature, std::vector<std::vector<float>>& HogFeature, std::vector<std::string>& labelVector, Mat& retMat, Mat& retLabels) {

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

	// Create a Mat
	Mat featureMatrix(Size(HuMomentfeature[0].size() + HogFeature[0].size(), labelVector.size()), CV_32F);
	retMat = featureMatrix;
	Mat labels(Size(1, labelVector.size()), CV_32F);
	retLabels = labels;
	int i, j, k;

	for (i = 0; i < featureMatrix.rows; i++) {

		for (j = 0; j < HuMomentfeature[i].size(); j++) {
			featureMatrix.at<float>(i, j) = HuMomentfeature[i][j];
		}

		for (k = 0; k < HogFeature[i].size(); k++, j++) {
			featureMatrix.at<float>(i, j) = HogFeature[i][k];
		}

		int val = 7;
		auto it = label_name.find(labelVector[i]);
		if (it != label_name.end())
			val = it->second;

		labels.at<float>(i) = val;
	}
}

int main() {

	cv::setNumThreads(0);
	system("cls");
	/*STEP 1 -> SEGMENTATION*/

	/*PREPROCESSING - Variable Declaration*/
	std::vector<Mat> furtherSegments;
	std::vector<Mat> firstSegments;
	std::vector<std::string> allLabels;
	std::vector<std::string> inputNames;
	std::vector<std::string> userInputs;

	// Reading the input paths file (user specified)
	ifstream inputPathsFile("Input Dataset/testInputs.txt");

	// Start Segmentation Process
	std::cout << "Begin Segmentation" << endl << endl;

	int ct = 0;

	if (inputPathsFile.is_open()) {
		std::string name_temp;
		while (getline(inputPathsFile, name_temp)) {
			userInputs.push_back(name_temp);
			segmentMain(name_temp, furtherSegments, allLabels, firstSegments, inputNames); // returns -> 1) Internal Segment 2) Encoded label 3) Initial Segment 4) Names of succesfully segmented
		}
	}

	for (int i = 0; i < userInputs.size(); i++) {
		string statusMsg;

		if (std::count(inputNames.begin(), inputNames.end(), userInputs[i])) {
			statusMsg = "Segmented Succesfully";
		}
		else {
			statusMsg = "Segmentation Failed (Bad Lighting Condition)";
		}

		cout << left << setw(3) << ++ct;
		cout << left << setw(35) << setfill(' ') << userInputs[i];
		cout << left << setw(35) << setfill(' ') << statusMsg << endl;
	}


	/*STEP 2 - Feature Extraction*/

	// Start Feature Extraction
	std::cout << "\nBegin Feature Extraction";

	// Feature Vector
	std::vector<std::vector<double>> huMomentsFeatures;
	std::vector<std::vector<float>> HoGFeatures;

	// Call feature extraction function
	extractFeaturesFromTrafficSign(furtherSegments, firstSegments, huMomentsFeatures, HoGFeatures);

	std::cout << "\t....Done" << endl << endl;


	/*STEP 3 - Writing Feature to Matrix*/

	std::cout << "Generating feature matrix";

	Mat features;
	Mat labels;

	generateFeatureMatrix(huMomentsFeatures, HoGFeatures, allLabels, features, labels);

	std::cout << "\t....Done" << endl << endl;


	/*STEP 5 - Predicting using on pretrained model*/
	
	std::cout << "Begin prediction";

	// Load Models
	Ptr<ml::SVM> pre_svm = Algorithm::load<ml::SVM>("Models/NoStd/T6G1_SVM.xml");

	Ptr<ml::RTrees> pre_rtrees = Algorithm::load<ml::RTrees>("Models/NoStd/T6G1_rtree.xml");

	// Predict (With Confidence)
	// SVM
	Mat svm_res;
	pre_svm->predict(features, svm_res);

	// Random Forest
	Mat rtree_res;
	Mat rtree_votes; 
	pre_rtrees->predict(features, rtree_res);
	pre_rtrees->getVotes(features, rtree_votes, 0);

	std::cout << "\t....Done" << endl << endl;


	/*STEP 6 - Analysis*/

	std::string signNames[] = { "No horn", "Car", "Uturn", "People crossing", "No stopping", "No entry", "Invalid Type" };

	cout << "\n===================================================================================" << endl;
	cout << "Predicted signs based on user input" << endl;
	cout << "===================================================================================" << endl << endl;
	cout << left << setw(34) << setfill(' ') << "Sign Input Name";
	cout << left << setw(18) << setfill(' ') << "Actual Label";
	cout << left << setw(18) << setfill(' ') << "SVM Prediction";
	cout << left << setw(20) << setfill(' ') << "Random Forest Prediction (Confidence)" << endl << endl;

	for (int i = 0; i < inputNames.size(); i++) {
		string curLabel;
		string svmPredLabel;
		string rtreePredLabel;
		double max = 0;

		Mat temp = imread(inputNames[i]);
		resize(temp, temp, Size(150, 150), cv::INTER_LINEAR);
		imshow("Current Image", temp);

		// Process Confidence
		Mat signProbasRow = rtree_votes.row(i + 1);
		minMaxLoc(signProbasRow, NULL, &max, NULL, NULL);

		// Rounding
		std::stringstream stream;
		stream << std::fixed << std::setprecision(2) << max;
		std::string probaVal = stream.str();
		
		// Prepping strings to be read
		curLabel = signNames[int(labels.at<float>(i)) - 1];
		svmPredLabel = signNames[int(svm_res.at<float>(i)) - 1];
		rtreePredLabel = signNames[int(rtree_res.at<float>(i)) - 1] + " (" + probaVal + "%)";

		// Print results
		cout << left << setw(34) << setfill(' ') << inputNames[i] + ":";
		cout << left << setw(18) << setfill(' ') << curLabel;
		cout << left << setw(18) << setfill(' ') << svmPredLabel;
		cout << setprecision(2) << left << setw(20) << setfill(' ') << rtreePredLabel;
		cout << endl << endl;

		waitKey(0);
	}

	return 0;
}