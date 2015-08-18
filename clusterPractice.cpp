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

Mat reshapeCol(Mat in){
  Mat points(in.rows*in.cols, 1,CV_32F);
  int cnt = 0;
  for(int i =0;i<in.rows;i++){
    for(int j=0;j<in.cols;j++){
      points.at<float>(cnt, 0) = in.at<Vec3b>(i,j)[0];
      cnt++;
    }
  }
  return points;
}


int main(){
  cout << "Start.." << endl;
  // Cluster the same image in a stack then on it's own and compare results

  Mat in = imread("../Lena.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat test1 = reshapeCol(in);

  int dictSize = 10;
  int attempts = 10;
  int flags = KMEANS_PP_CENTERS;
  TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);

  BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);
  bowTrainer.add(test1);
  Mat output = bowTrainer.cluster();
  cout << "These are the clusters: " << output << endl;

  BOWKMeansTrainer bowTrainer1(dictSize, tc, attempts, flags);
  bowTrainer1.add(test1);
  Mat output1 = bowTrainer1.cluster();
  cout << "These are the clusters: " << output1 << endl;

  return 0;
}
