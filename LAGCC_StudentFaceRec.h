// Copyright(c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
//Released to public domain under terms of the BSD Simplified license.

//Modified to suit the project

#include <iostream>
#include <string>
#include <vector>
#include <Windows.h>

//include opencv core
#include "opencv2\core\core.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"

//file handling
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

static Mat MatNorm(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

//Function to read the selected LAGCC students' face dataset file gathered
static void readStudentDB_file(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file) {
		string error = "no valid input file";
		CV_Error(CV_StsBadArg, error);
	}

	string line, path, label;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, label);
		if (!path.empty() && !label.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(label.c_str()));
		}
	}
}

//Function to train the selected LAGCC students' face dataset 
void fisherFaceTrainer() {

	// These two vectors holds the images and labels for training
	vector<Mat> images; //Vector of class Mat to store the images
	vector<int> labels; //Vector to store labels - Think of the label as the subject(the student) this image 
						//belongs to, so same subjects(students) should have the same label.

	//This file has the database ready to be trained into the system
	string filename = "C:/faces/faces.txt";

	try {
		readStudentDB_file(filename, images, labels); //Function call to read the images and their labels

		cout << "Training 4 LAGCC students face data into the system\n";
		cout << "size of the images is " << images.size() << endl; //outputs the total number of images in the database
		cout << "size of the labels is " << labels.size() << endl; //outputs the number of labels
		cout << "Training begins...." << endl << endl;
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	Ptr<FaceRecognizer> model = createFisherFaceRecognizer(40);

	//trains the input dataset
	model->train(images, labels);

	int height = images[0].rows;

	//saves the trained dataset
	model->save("C:/FDB/yaml/studentfacedata.yml");

	cout << "Training finished...." << endl << endl;

	waitKey(10);
}

//Function that detects and recognizes the student
int  LAGCC_StudentsFaceRecognition() {

	cout << "start recognizing..." << endl;

	//load pre-trained datasets
	Ptr<FaceRecognizer>  model = createFisherFaceRecognizer(40);

	model->load("C:/FDB/yaml/studentfacedata.yml");

	Mat testSample = imread("C:/db/student1/elija1.jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;

	//String holds the haar cascade frontal face file
	string classifier = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";

	//Haar Cascade Classifier used purposefully for facial detection
	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(classifier)) {
		cout << " Error loading file" << endl;
		return -1;
	}

	VideoCapture cap(0); //To capture live feed from the webcam

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return -1;
	}

	namedWindow(window, 1);
	long count = 0;

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat grayScaleFrame;
		Mat original;

		cap >> frame;
		count = count + 1;//count frames;

		if (!frame.empty()) {

			//clone from original frame
			original = frame.clone();

			//convert image to gray scale and equalize
			cvtColor(original, grayScaleFrame, CV_BGR2GRAY);
			equalizeHist(grayScaleFrame, grayScaleFrame);

			//detect face in gray image
			face_cascade.detectMultiScale(grayScaleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			//number of faces detected
			cout << faces.size() << " faces detected" << endl;
			string frameset = to_string(count);
			string faceset = to_string(faces.size());

			int width = 0, height = 0;

			for (int i = 0; i < faces.size(); i++)
			{
				//region of interest
				Rect face_i = faces[i];

				//crop the roi from gray image
				Mat face = grayScaleFrame(face_i);

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				//recognizing what faces detected
				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				//cout << " confidence " << confidence << endl;

				string text; //string to hold whether the person detected is a student or not

				if (label == 0) {

					//draws a GREEN rectangle in the recognized or identified student's face
					rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
					text = "Student Identified";
				}
				else {

					//draws a RED rectangle in th]]le unrecognized or unidentified person's face
					rectangle(original, face_i, CV_RGB(255, 0, 0), 1);
					text = "Unidentified person";
					//plays alert sound when an unidentified person is detected
					PlaySound("C:\\faces\\unidentifiedpersonsound.wav", NULL, SND_ASYNC);
				}

				int pos_x = max(face_i.tl().x - 10, 0);
				int pos_y = max(face_i.tl().y - 10, 0);

				//Outputs whether the person detected is a student or not
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			}

			//outputs the frameset
			putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			cv::imshow(window, original);
		}

		if (waitKey(30) >= 0)
			break;
	}


}
