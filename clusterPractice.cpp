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

using namespace cv;
using namespace std;


int main(){
  cout << "Start.." << endl;

  VideoCapture cap;
  cap.open("../testVid.mp4");
  if(!cap.isOpened()){
    cout << "video unable to be opened. Exiting.." << endl;
    exit(1);
  }


  vector<Mat> store;
  Mat img;
  namedWindow("CurrentImg", CV_WINDOW_AUTOSIZE);
  namedWindow("FrameCount", CV_WINDOW_AUTOSIZE);
  int w_width = 200;
  int w_height = 75;

  for(int i=0;i<cap.get(CV_CAP_PROP_FRAME_COUNT);i++){
    cap >> img;
    Mat frame = Mat(w_height, w_width, CV_8UC3, Scalar(255,255,255));

    double fnum = cap.get(CV_CAP_PROP_POS_FRAMES);

    stringstream ss;
    ss << "Frame: " << fnum;
    Size textsize = getTextSize(ss.str(), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 2, 0);
    Point org((w_width - textsize.width)/2, (w_height - textsize.height)/2+20);
    putText(frame, ss.str(), org, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 2, 8 );

    imshow("FrameCount", frame);
    imshow("CurrentImg", img);
    cout << "This is the size: " << img.size() << " and i count: " << i << endl;
    char c = waitKey(30000);
    if(c=='s'){
      store.push_back(img.clone());
    }else if(c=='q'){
      break;
    }
 }

  cout << "what is the name of that class? " << endl;
  string clsnme;
  cin >> clsnme;
  cout << "This was the class name: " << clsnme << endl;

 for(int i=0;i<store.size();i++){
   stringstream ss;
   ss << "../Images/";
   ss << clsnme << '_' << i << ".jpg";
   imwrite(ss.str(), store[i]);
   cout << "written: " << ss.str() << endl;
 }

  return 0;
}
