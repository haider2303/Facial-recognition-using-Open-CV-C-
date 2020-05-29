#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "preprocessFace.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

// Global Variables And Constants
string fn_csv = string("ellipse_training_data.csv");
const int faceWidth = 200;
const int faceHeight = faceWidth;
// Cascade Classifier file, used for Face Detection.
const char *faceCascadeFilename = "Z:\\Caascades\\lbpcascade_frontalface.xml";     // LBP face detector. // 19/40 // Detected
//const char *faceCascadeFilename = "Z:\\Caascades\\haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *faceCascadeFilename = "Z:\\Caascades\\haarcascade_frontalface_alt.xml";  // Haar face detector.
//const char *faceCascadeFilename = "Z:\\Caascades\\haarcascade_frontalface_default.xml";  // Haar face detector.
//const char *faceCascadeFilename = "Z:\\Caascades\\haarcascade_frontalface_alt2.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "Z:\\Caascades\\haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "Z:\\Caascades\\haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.
//int globalcount = 0; // Will be used as a global variable to analyse a function's behavior.

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, std::map<int, string>& labelsInfo, char separator = ';')
{
	ifstream csv(filename.c_str());
	if (!csv) CV_Error(CV_StsBadArg, "No valid input file was given, please check the given filename.");
	string line, path, classlabel, info;
	while (getline(csv, line)) {
		stringstream liness(line);
		path.clear(); classlabel.clear(); info.clear();
		getline(liness, path, separator);
		getline(liness, classlabel, separator);
		getline(liness, info, separator);
		if (!path.empty() && !classlabel.empty()) {
			////cout << "Processing " << path << endl;
			int label = atoi(classlabel.c_str());
			if (!info.empty())
				labelsInfo.insert(std::make_pair(label, info));
			// 'path' can be file, dir or wildcard path
			String root(path.c_str());
			vector<String> files;
			glob(root, files, true);
			for (vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) {
				//cout << "\t" << *f << endl;
				Mat img = imread(*f, CV_LOAD_IMAGE_GRAYSCALE);
				static int w = -1, h = -1;
				static bool showSmallSizeWarning = true;
				if (w>0 && h>0 && (w != img.cols || h != img.rows)) cout << "\t* Warning: images should be of the same size!" << endl;
				if (showSmallSizeWarning && (img.cols<50 || img.rows<50)) {
					cout << "* Warning: for better results images should be not smaller than 50x50!" << endl;
					showSmallSizeWarning = false;
				}
				images.push_back(img);
				labels.push_back(label);
			}
		}
	}
}

void read_test_images(const string& filename, vector<Mat>& test_images, char separator = ';')
{
	ifstream csv(filename.c_str());
	if (!csv) CV_Error(CV_StsBadArg, "No valid input file was given, please check the given filename.");
	string line, path, classlabel, info;

	while (getline(csv, line)) 
	{
		stringstream liness(line);
		path.clear(); classlabel.clear(); info.clear();
		getline(liness, path, separator);

		if (!path.empty())
		{
			cout << "Processing " << path << endl;

			// 'path' can be file, dir or wildcard path
			String root(path.c_str());
			vector<String> files;
			glob(root, files, true);

			for (vector<String>::const_iterator f = files.begin(); f != files.end(); ++f) 
			{
				cout << "\t" << *f << endl;

				Mat test_img = imread(*f);

				/*int myInt = globalcount;
				globalcount++;*/

				/*stringstream myString;
				myString << myInt << ".png";
				imwrite(myString.str(), test_img);*/

				if (test_img.rows != 0)
				{
					test_images.push_back(test_img);
				}
				
			}
		}
	}

}

void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
	// Load the Face Detection cascade classifier xml file.
	try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
		faceCascade.load(faceCascadeFilename);
	}
	catch (cv::Exception &e) {}
	if (faceCascade.empty()) {
		cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
		cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
		exit(1);
	}
	cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;

	// Load the Eye Detection cascade classifier xml file.
	try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
		eyeCascade1.load(eyeCascadeFilename1);
	}
	catch (cv::Exception &e) {}
	if (eyeCascade1.empty()) {
		cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
		cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\haarcascades') into this WebcamFaceRec folder." << endl;
		exit(1);
	}
	cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;

	// Load the Eye Detection cascade classifier xml file.
	try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
		eyeCascade2.load(eyeCascadeFilename2);
	}
	catch (cv::Exception &e) {}
	if (eyeCascade2.empty()) {
		cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
		// Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
		//exit(1);
	}
	else
		cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}

int reChecking(Mat unknownimg);

int main(int argc, const char *argv[]) 
{   
	CascadeClassifier faceCascade;
	CascadeClassifier eyeCascade1;
	CascadeClassifier eyeCascade2;
	initDetectors(faceCascade, eyeCascade1, eyeCascade2);


    // These vectors hold the images and corresponding labels:
    vector<Mat> training_images;
    vector<int> labels;
	std::map<int, std::string> labelsInfo;
    

    try 
	{
        read_csv(fn_csv, training_images, labels, labelsInfo);
    }
	catch (cv::Exception& e) 
	{
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		getchar();
        exit(1);
    }
    

	vector<Mat> test_images; // -> Will contain the test images given by Dr.Faisal

	try
	{
		read_test_images("testing_data_path.csv", test_images, ';');

	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}


    // Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->setLabelsInfo(labelsInfo);
    model->train(training_images, labels);


	ofstream write_file("Output.csv");

	string filename = " ";
	string personName = " ";
	int prediction = -1;

	for (unsigned int i = 0, j = 0; i < test_images.size(); i++)
	{
		filename.clear();
		personName.clear();

		 //Find a face and preprocess it to have a standard size and contrast & brightness.
		Rect faceRect;  // Position of detected face.
		Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
		Point leftEye, rightEye;    // Position of the detected eyes.
		Mat face_resized = getPreprocessedFace(test_images[i], faceWidth, faceCascade, eyeCascade1, eyeCascade2, false, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

		if (face_resized.rows != 0)
		{
			try
			{
				int myInt = i;
				stringstream myString;
				myString << myInt << ".png";

				try
				{
					imwrite(myString.str(), face_resized);
				}
				catch (exception &d)
				{
					cerr << "imwrite exception thrown !" << endl;
				}

				prediction = model->predict(face_resized); // -> It all comes down to this.

			}
			catch (exception &e)
			{

				cerr << e.what() << endl;
				system("pause");
			}

			personName = model->getLabelInfo(prediction);

			cout << "Predicted Label: " << prediction << " File Name: " << i << " - Person Name: " << personName << endl;

		}
		else
		{

			prediction = reChecking(test_images[i]);
			personName = model->getLabelInfo(prediction);
			cout << "Predicted Label: " << prediction << " File Name: " << i << " - Person Name: " << personName << endl;

		}

		write_file << i << ".png" << "," << "traindata" << prediction << endl;

	}

	write_file.close();
	getchar();
    return 0;
}


int reChecking(Mat unknownimg)
{
	string fn_haar = "haarcascade_frontalface_alt.xml";
	string fn_csv = "csv.csv";
	
	vector<Mat> images;
	vector<int> labels;
	std::map<int, std::string> labelsInfo;

	try 
	{
		read_csv(fn_csv, images, labels, labelsInfo);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}

	int im_width = images[0].cols;
	int im_height = images[0].rows;

	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);
	
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);

	
	
	Mat frame;
	frame = unknownimg;

	Mat original = frame.clone();
	Mat gray;
	if (original.empty())
		system("pause");
	else if (original.channels() > 1)
		cvtColor(original, gray, CV_BGR2GRAY);
	else gray = original;

	vector< Rect_<int> > faces;
	haar_cascade.detectMultiScale(gray, faces);

	int prediction = -1;

	for (int i = 0; i < faces.size(); ) 
	{
		// Process face by face:
		Rect face_i = faces[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = gray(face_i);
		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
	 prediction = model->predict(face_resized);
		rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
		break;
	}
	
	return prediction;

}
