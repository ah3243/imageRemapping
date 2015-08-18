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

  int numIterations = 5;


  int attempts = 10;
  int flags = KMEANS_PP_CENTERS;
  TermCriteria tc(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.0001);

  vector<vector<int> > tRate;
  vector<int> a;
  tRate.push_back(a);
  Mat in = imread("../AFoil_1.png", CV_LOAD_IMAGE_GRAYSCALE);

  const int channels = 0;
  const int histSize = 10;
  float inner[2] = {0, 10000};
  const float* holder= {inner};

  Mat test1 = reshapeCol(in);
  for(int j=1;j<10;j++){
    tRate.push_back(a);
    cout << "Finding: " << j << " Clusters." << endl;
    int dictSize = j;
    for(int i=0;i<numIterations;i++){
      BOWKMeansTrainer bowTrainer(dictSize, tc, attempts, flags);
      BOWKMeansTrainer bowTrainer1(dictSize, tc, attempts, flags);

      bowTrainer.add(test1);
      bowTrainer.add(test1);
      bowTrainer.add(test1);
      bowTrainer.add(test1);
      bowTrainer.add(test1);
      Mat output = bowTrainer.cluster();
      bowTrainer.clear();

      bowTrainer1.add(test1);
      Mat output1 = bowTrainer1.cluster();
      bowTrainer1.clear();

      Mat out, out1;
      calcHist(&output, 1, &channels, Mat(), out, 1, &histSize, &holder, true, false);
      calcHist(&output1, 1, &channels, Mat(), out1, 1, &histSize, &holder, true, false);

      double distance = compareHist(out, out1,CV_COMP_CHISQR);

      cout << "This is the similarity: " << distance << endl;
      if(distance == 0){
        tRate[j].push_back(1);
      }
    }
  }
  return 0;
}
