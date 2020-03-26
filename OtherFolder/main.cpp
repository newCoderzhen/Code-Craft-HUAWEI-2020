#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <iomanip>
using namespace std;

#define TEST ture;

struct Data {
    vector<double> features;
    int label;
    Data(vector<double> f, int l) : features(f), label(l)
    {}
};
struct Param {
    vector<vector<double>> wtSet;
};


class LR {
public:
    void train();
    void predict();
    int loadModel();
    int storeModel();
    LR(string trainFile, string testFile, string predictOutFile);

private:
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    vector<int> predictVec;
    Param param;
    Param param2;
    string trainFile;
    string testFile;
    string predictOutFile;
    string weightParamFile = "modelweight.txt";

private:
    bool init();
    bool loadTrainData();
    bool loadTestData();
    int storePredict(vector<int> &predict);
    void initParam();
    double feedforward(const Data &data, vector<double>& out1);
    double reluCalc(const double wxb);
    double reluPraim(const double wxb);
    double sigmoidCalc(const double wxb);
    double lossCal(vector<double>& sigmoidVec);
    void gradientSlope(const vector<Data> &dataSet, int j, const vector<double> &sigmoidVec, const vector<vector<double>>& hiddenOuts);

private:
    int featuresNum;
    const int layer1_size = 30;
    const int layer2_size = 1;
    const double wtInitV = 1.0;
    const double stepSize = 0.35;
    const int maxIterTimes = 10;
    const double predictTrueThresh = 0.5;
    const int train_show_step = 1;
    const int batch_size = 200;
};

LR::LR(string trainF, string testF, string predictOutF)
{
    trainFile = trainF;
    testFile = testF;
    predictOutFile = predictOutF;
    featuresNum = 0;
    init();
}

bool LR::loadTrainData()
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

void LR::initParam()
{
    vector<double> temp1(featuresNum, wtInitV);
    vector<double> temp2(layer1_size, wtInitV);
    for (int i = 0; i < layer1_size; i++) {
        param.wtSet.push_back(temp1);
    }
    for (int i = 0; i < layer2_size; i++) {
        param2.wtSet.push_back(temp2);
    }
}

bool LR::init()
{
    trainDataSet.clear();
    bool status = loadTrainData();
    if (status != true) {
        return false;
    }
    featuresNum = trainDataSet[0].features.size();
    param.wtSet.clear();
    param2.wtSet.clear();
    initParam();
    return true;
}


double LR::feedforward(const Data &data, vector<double>& out1)
{
    double wtv, feav;
    
    for(int i = 0; i < param.wtSet.size(); ++i) {
        double mulSum = 0.0L;
        for(int j = 0; j < param.wtSet[0].size(); ++j) {
            wtv = param.wtSet[i][j];
            feav = data.features[j];
            mulSum += wtv * feav;
        }
        out1.push_back(mulSum);
    }
    double mulSum = 0.0L;
    for(int j = 0; j < param2.wtSet[0].size(); ++j) {
        wtv = param2.wtSet[0][j];
        feav = reluCalc(out1[j]);
        mulSum += wtv * feav;
    }
    return mulSum;
}


double LR::reluCalc(const double wxb) 
{
    return max(0.0, wxb);
}


double LR::reluPraim(const double wxb) {
    return wxb > 0.0 ? 1.0 : 0.0;
}


inline double LR::sigmoidCalc(const double wxb)
{
    double expv = exp(-1 * wxb);
    double expvInv = 1 / (1 + expv);
    return expvInv;
}


double LR::lossCal(vector<double>& sigmoidVec)
{
    double lossV = 0.0L;
    int i;

    for (i = 0; i < trainDataSet.size(); i++) {
        lossV -= trainDataSet[i].label * log(sigmoidVec[i]);
        lossV -= (1 - trainDataSet[i].label) * log(1 - sigmoidVec[i]);
    }
    lossV /= trainDataSet.size();
    return lossV;
}


void LR::gradientSlope(const vector<Data> &dataSet, int j, const vector<double> &sigmoidVec, const vector<vector<double>>& hiddenOuts)
{
    double sigv, label, temp;
    for(int index = 0; index < param2.wtSet[0].size(); ++index) {
        double gsV = 0.0L;
        vector<double> gsV2(featuresNum, 0.0L);
        for (int i = j - batch_size; i < j; i++) {
            sigv = sigmoidVec[i];
            label = dataSet[i].label;
            temp = (label - sigv) * reluCalc(hiddenOuts[i][index]);
            gsV += temp;
            for(int k = 0; k < featuresNum; ++k) {
                gsV2[k] += (label - sigv) * param2.wtSet[0][index] * reluPraim(hiddenOuts[i][index]) * dataSet[i].features[k];
            }
        }
        gsV = gsV / batch_size;
        param2.wtSet[0][index] += stepSize * gsV;
        for(int idx = 0; idx < featuresNum; ++idx) {
            param.wtSet[index][idx] += stepSize * gsV2[idx] / batch_size;
        }
    }
}

void LR::train()
{
    double sigmoidVal;
    double wxbVal;
    int i, j;

    for (i = 0; i < maxIterTimes; i++) {
        vector<double> sigmoidVec;
        vector<vector<double>> hiddenOuts;

        for (j = 0; j < trainDataSet.size(); j++) {
            vector<double> out1;
            wxbVal = feedforward(trainDataSet[j], out1);
            sigmoidVal = sigmoidCalc(wxbVal);
            hiddenOuts.push_back(out1);
            sigmoidVec.push_back(sigmoidVal);
            if(j % batch_size == 0 && j != 0) {
                gradientSlope(trainDataSet, j, sigmoidVec, hiddenOuts);
            }
        }
        // Update parameters: 
        // for (j = 0; j < param.wtSet.size(); j++) {
        //     param.wtSet[j] += stepSize * gradientSlope(trainDataSet, j, sigmoidVec);
        // }

        if (i % train_show_step == 0) {
            cout << "iter " << i << ". Loss is : " << lossCal(sigmoidVec) << endl;
        }
    }
}

void LR::predict()
{
    double sigVal;
    int predictVal;
    vector<double> out1;

    loadTestData();
    for (int j = 0; j < testDataSet.size(); j++) {
        sigVal = sigmoidCalc(feedforward(testDataSet[j], out1));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        predictVec.push_back(predictVal);
    }

    storePredict(predictVec);
}

int LR::loadModel()
{
    string line;
    double dbt;

    ifstream fin(weightParamFile.c_str());
    if (!fin) {
        cout << "打开模型参数文件失败" << endl;
        exit(0);
    }

    getline(fin, line);
    stringstream sin(line);
    for(int i = 0; i < param.wtSet.size(); ++i) {
        vector<double> wtTmp;
        for (int j = 0; j < featuresNum; ++j) {
            char c = sin.peek();
            if (c == -1) {
                cout << "模型参数数量少于特征数量，退出" << endl;
                return -1;
            }
            sin >> dbt;
            wtTmp.push_back(dbt);
        }
        param.wtSet[i].swap(wtTmp);
    }
    vector<double> wtTmp;
    for (int j = 0; j < param2.wtSet[0].size(); ++j) {
        char c = sin.peek();
        if (c == -1) {
            cout << "模型参数数量少于特征数量，退出" << endl;
            return -1;
        }
        sin >> dbt;
        wtTmp.push_back(dbt);
    }
    param.wtSet[0].swap(wtTmp);
    fin.close();
    return 0;
}

int LR::storeModel()
{
    string line;

    ofstream fout(weightParamFile.c_str());
    if (!fout.is_open()) {
        cout << "打开模型参数文件失败" << endl;
    }
    if (param.wtSet[0].size() < featuresNum) {
        cout << "wtSet1 size is " << param.wtSet.size() << endl;
    }
    if (param2.wtSet[0].size() < layer1_size) {
        cout << "wtSet2 size is " << param2.wtSet[0].size() << endl;
    }
    for(int i = 0; i < param.wtSet.size(); ++i) {
        for (int j = 0; j < featuresNum; ++j) {
            fout << param.wtSet[i][j] << " ";
        }
    }
    for(int i = 0; i < param2.wtSet[0].size(); ++i) {
        fout << param2.wtSet[0][i] << " ";
    }
    fout.close();
    return 0;
}


bool LR::loadTestData()
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

int LR::storePredict(vector<int> &predict)
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

    LR logist(trainFile, testFile, predictFile);

    cout << "ready to train model" << endl;

    time_t start1, end1;
    time(&start1);
    logist.train();
    time(&end1);
    cout << "Training Time: " << double(end1 - start1) << setprecision(5) << endl;

    cout << "training ends, ready to store the model" << endl;
    logist.storeModel();

#ifdef TEST
    cout << "ready to load answer data" << endl;
    loadAnswerData(answerFile, answerVec);
#endif

    cout << "let's have a prediction test" << endl;
    logist.predict();

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
