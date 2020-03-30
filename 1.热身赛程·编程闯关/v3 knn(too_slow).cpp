#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <iomanip>
#include <algorithm>
using namespace std;

#define TEST true;

struct Data {
    vector<double> features;
    int label;
    double dist;
    Data(vector<double> f, int l) : features(f), label(l), dist(0.0)
    {}
};

bool comparison(Data a, Data b) 
{ 
    return (a.dist < b.dist); 
} 

class KNN {
public:
    void train();
    void predict();
    KNN(string trainFile, string testFile, string predictOutFile);

private:
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    vector<int> predictVec;
    string trainFile;
    string testFile;
    string predictOutFile;

private:
    bool init();
    bool loadTrainData();
    bool loadTestData();
    int storePredict(vector<int> &predict);
    double distanceCalc(vector<double>& test, vector<double>& train);

private:
    int featuresNum;
};

KNN::KNN(string trainF, string testF, string predictOutF)
{
    trainFile = trainF;
    testFile = testF;
    predictOutFile = predictOutF;
    featuresNum = 0;
    init();
}

bool KNN::loadTrainData()
{
    ifstream infile(trainFile.c_str());
    string line;

    if (!infile) {
        cout << "打开训练文件失败" << endl;
        exit(0);
    }

    while (infile) {
        getline(infile, line);
        if (line.size() > featuresNum) {
            stringstream sin(line);
            char ch;
            double dataV;
            int i;
            vector<double> feature;
            i = 0;

            while (sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "训练文件数据格式不正确，出错行为" << (trainDataSet.size() + 1) << "行" << endl;
                    return false;
                }
            }
            int ftf;
            ftf = (int)feature.back();
            feature.pop_back();
            trainDataSet.push_back(Data(feature, ftf));
        }
    }
    infile.close();
    return true;
}

bool KNN::init()
{
    trainDataSet.clear();
    bool status = loadTrainData();
    if (status != true) {
        return false;
    }
    featuresNum = trainDataSet[0].features.size();
    return true;
}

double KNN::distanceCalc(vector<double>& test, vector<double>& train)
{
    double dist = 0.0;
    for(int i = 0; i < featuresNum; ++i) {
        dist += (test[i] - train[i]) * (test[i] - train[i]);
    }
    return sqrt(dist);
}

void KNN::predict()
{
    int predictVal;

    loadTestData();
    for (int j = 0; j < testDataSet.size(); j++) {
        for(int i = 0; i < trainDataSet.size(); ++i) {
            trainDataSet[i].dist = distanceCalc(testDataSet[j].features, trainDataSet[i].features);
        }
        if(j % 100 == 0)
            cout << j << endl;
        sort(trainDataSet.begin(), trainDataSet.end(), comparison);
        int freq1 = 0;     // Frequency of group 0 
        int freq2 = 0;     // Frequency of group 1 
        for (int i = 0; i < 100; i++) 
        { 
            if (trainDataSet[i].label == 0) 
                freq1++; 
            else if (trainDataSet[i].label == 1) 
                freq2++; 
        } 
        predictVal = freq1 > freq2 ? 0 : 1;
        predictVec.push_back(predictVal);
    }

    storePredict(predictVec);
}

bool KNN::loadTestData()
{
    ifstream infile(testFile.c_str());
    string lineTitle;

    if (!infile) {
        cout << "打开测试文件失败" << endl;
        exit(0);
    }

    while (infile) {
        vector<double> feature;
        string line;
        getline(infile, line);
        if (line.size() > featuresNum) {
            stringstream sin(line);
            double dataV;
            int i;
            char ch;
            i = 0;
            while (i < featuresNum && sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "测试文件数据格式不正确" << endl;
                    return false;
                }
            }
            testDataSet.push_back(Data(feature, 0));
        }
    }

    infile.close();
    return true;
}

bool loadAnswerData(string awFile, vector<int> &awVec)
{
    ifstream infile(awFile.c_str());
    if (!infile) {
        cout << "打开答案文件失败" << endl;
        exit(0);
    }

    while (infile) {
        string line;
        int aw;
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            sin >> aw;
            awVec.push_back(aw);
        }
    }

    infile.close();
    return true;
}

int KNN::storePredict(vector<int> &predict)
{
    string line;
    int i;

    ofstream fout(predictOutFile.c_str());
    if (!fout.is_open()) {
        cout << "打开预测结果文件失败" << endl;
    }
    for (i = 0; i < predict.size(); i++) {
        fout << predict[i] << endl;
    }
    fout.close();
    return 0;
}

int main(int argc, char *argv[])
{
    vector<int> answerVec;
    vector<int> predictVec;
    int correctCount;
    double accurate;
    string trainFile = "./data/train_data.txt";
    string testFile = "./data/test_data.txt";
    string predictFile = "./projects/student/result.txt";

    string answerFile = "./projects/student/answer.txt";

    KNN knn(trainFile, testFile, predictFile);

    cout << "ready to directly predict" << endl;

    time_t start1, end1;
    time(&start1);
    knn.predict();
    time(&end1);
    cout << "Costing Time: " << setprecision(5) << double(end1 - start1) << endl;

#ifdef TEST
    cout << "ready to load answer data" << endl;
    loadAnswerData(answerFile, answerVec);
#endif

    cout << "let's have a prediction test" << endl;
    // knn.predict();

#ifdef TEST
    loadAnswerData(predictFile, predictVec);
    cout << "test data set size is " << predictVec.size() << endl;
    correctCount = 0;
    for (int j = 0; j < predictVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == predictVec[j]) {
                correctCount++;
            }
        } else {
            cout << "answer size less than the real predicted value" << endl;
        }
    }

    accurate = ((double)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
#endif

    return 0;
}
