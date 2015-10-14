// Practice for Clustering with Opencv BOW classes

#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <iostream> // General io
#include <stdlib.h>
#include <fstream>
#include <string>
#include <algorithm> // Maybe fix DescriptorExtractor doesn't have a member 'create'
#include <boost/filesystem.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

// Check for input dir, exit if none
void checkInDir(string inDir){
  if(!boost::filesystem::exists(inDir) && boost::filesystem::is_directory(inDir)){
    cout << "Input image dir not able to be found. Exiting."<< endl;
    exit(1);
  }
}
// Check for Dir if not exists create it
void checkImgDirs(string outDir){
  if(!boost::filesystem::exists(outDir)){
    if(boost::filesystem::create_directory(outDir)){
      cout << "Directory Created: " << outDir << endl;
    }else{
      cout << "Directory creation failed" << outDir << endl;
    }
  }else{
    cout << "Output Dir Already Exists" << endl;
  }
}

// Extract Class Names from Dir Paths
void extractClsNmes(vector<string> paths, vector<string> &clsNmes){
  for(int j=0;j<paths.size();j++){
    size_t found = paths[j].find_last_of('/');
    clsNmes.push_back(paths[j].substr(found+1));
  }
}

// Get all Dirs at path
void getDirNmes(string inDir, vector<string>& clsPaths){
  boost::filesystem::directory_iterator end_iter;

  for( boost::filesystem::directory_iterator dir_iter(inDir) ; dir_iter != end_iter ; ++dir_iter)
  {
    if (boost::filesystem::is_directory(*dir_iter) )
    {
      clsPaths.push_back(dir_iter->path().string());
    }
  }
}

// Generates Individual output Class Folders if they dont exist
void checkClsoutDirs(string dir, vector<string> clsNmes){
  for(int j=0;j<clsNmes.size();j++){
    stringstream ss;
    ss << dir << "/" << clsNmes[j];
    checkImgDirs(ss.str());
  }
}

int main(int argc, char** argv){
  string inDir = "/home/james-tt/Desktop/TEST_IMAGES/ARImgs/BACKUP";
  string outDir = "/home/james-tt/Desktop/TEST_IMAGES/ARImgs/Normalised";
  vector<string> clsPaths;
  vector<string> clsNmes;
  checkInDir(inDir); // Check for input img Dir

  getDirNmes(inDir, clsPaths);
  extractClsNmes(clsPaths, clsNmes);

  cout << "This is the size: " << clsPaths.size() << endl;
  for(int i=0;i<clsPaths.size();i++){
    cout << "ClassPaths: " << clsPaths[i] << endl;
    cout << "Classes: " << clsNmes[i] << endl;
  }
  checkImgDirs(outDir); // Check for output Dir
  checkClsoutDirs(outDir, clsNmes); // Check for individual Class Ouput Dirs

  // Create variables
  Mat cameraMatrix, distCoeffs, map1, map2;
  double image_Width, image_Height;

  // Get Distortion Coefficients and Camera Matrix from file
  cout << "Start.." << endl;
  FileStorage fs("../calFile.xml", FileStorage::READ);
  fs["Camera_Matrix"] >> cameraMatrix;
  fs["Distortion_Coefficients"] >> distCoeffs;
  fs["image_Width"] >> image_Width;
  fs["image_Height"] >> image_Height;
  fs.release();
  // Set size
  Size imageSize(image_Width, image_Height);

  // Create maps
  initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
      getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0, imageSize, 0),
      imageSize, CV_16SC2, map1, map2);

  // Create remap variables
  Mat view, rview;
  string fileNme = "/board_";
  string extension = ".jpg";

  // Create display Windows
  namedWindow("before", 0);
  namedWindow("after", 0);

  // Loop through imgs in dir
  for(int i=1;i<=15;i++){
    stringstream ss;
    ss << inDir << fileNme << i << extension;
    cout << "This is the inDir: " << ss.str() << endl;
    view = imread(ss.str(), 1);
    cout << "this is the size: " << view.size() << endl;
    imshow("before", view);
    // Remap image
   remap(view, rview, map1, map2, INTER_LINEAR);
   imshow("after", rview);
   ss.str("");
   ss << outDir << fileNme << i << extension;
   imwrite(ss.str(), rview);

    char c = waitKey(500);
    if(c=='q'){
      break;
    }
  }



  return 0;
}
