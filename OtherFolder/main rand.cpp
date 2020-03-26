#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <random>
#include <unistd.h>
#include<numeric>
using namespace std;
#define TEST true
#define Num  8001

struct Data {
    vector<double> features;
    int label;
    Data(vector<double> f, int l) : features(f), label(l)
    {}
};
struct Param {
    vector<double> wtSet;
};


class LR {
public:
	void train();
	void predict();
    void acurracy();
	int loadModel();
	int storeModel();
	LR(string trainFile, string testFile, string predictOutFile);

private:
	vector<Data> trainDataSet;
	vector<Data> testDataSet;
	vector<int> predictVec;
	Param param;
	string trainFile;
	string testFile;
	string predictOutFile;
	string weightParamFile = "modelweight.txt";

private:
	bool init();
	bool loadTrainData();
	bool loadTestData();
	int storePredict(vector<int>& predict);
	void initParam();
	double wxbCalc(const Data& data);
	double sigmoidCalc(const double& wxb);
	double lossCal();
    bool normalset(vector<Data>& dataset);
	double gradientSlope(const vector<Data>& dataSet, int index, const vector<double>& sigmoidVec,
		const vector<int>& indexes);

private:
	int featuresNum;
	const double wtInitV = 0.3;     //参数取消
    const double lambda =0.18;
	const double stepSize = 0.11;
	const int maxIterTimes = 10;
	const double predictTrueThresh = 0.678;
	const int train_show_step = 1000;
	const double batch = 0.105;
    const bool LOSS = true;
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
    auto start2 = chrono::steady_clock::now();
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
            feature.reserve(1001);
            i = 0;

            while (sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "训练文件数据格式不正确，出错行为." << (trainDataSet.size() + 1) << "行." << endl;
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
    auto end2 = chrono::steady_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    cout<< "读取数据time:. "  << elapsed2.count() << "s" << endl;
    return true;
}

bool LR::normalset(vector<Data>& dataset) {
	vector<float> sum, sd;
	int dim = dataset[0].features.size();
	int num = dataset.size();
	sum.reserve(dim + 1);
	sd.reserve(dim + 1);
	float sumtem = 0, sdtem = 0;
	for (int i = 0; i <dim; i++) {
		sumtem = 0;
		for (int j = 0; j < num; j++) {
			sumtem += dataset[j].features[i];
		}
		sum.push_back((sumtem / num));	   
	}
	for (int i = 0; i < dim; i++) {
		sdtem = 0;
		for (int j = 0; j < num; j++) {
			sdtem += (dataset[j].features[i]-sum[i])* (dataset[j].features[i] - sum[i]);
		}
		sdtem = sdtem / (num - 1);
		sd.push_back(sqrtf(sdtem));
	}
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < num; j++) {
			dataset[j].features[i]= (dataset[j].features[i]-sum[i])/sd[i];
		}
	}

	return true;
}
void LR::initParam()
{
    int i;
    double k;
    srand(1);
    for (i = 0; i < featuresNum; i++) {
        k = rand() % ( 100) / (float)(101);
        param.wtSet.push_back(k*0.01);
    }
}

bool LR::init()
{
    trainDataSet.clear();
    trainDataSet.reserve(Num);
    bool status = loadTrainData();
    if (status != true) {
        return false;
    }
    normalset( trainDataSet);
    featuresNum = trainDataSet[0].features.size();
    param.wtSet.clear();
    param.wtSet.reserve(1001);
    initParam();
    return true;
}


double LR::wxbCalc(const Data &data)
{
    double mulSum = 0.0L;
    int i;
    double wtv, feav;
    for (i = 0; i < param.wtSet.size(); i++) {
        wtv = param.wtSet[i];
        feav = data.features[i];
        mulSum += wtv * feav;
    }

    return mulSum;
}

inline double LR::sigmoidCalc(const double &wxb)
{
    double expv = exp(-1 * wxb);
    double expvInv = 1 / (1 + expv);
    return expvInv;
}


double LR::lossCal()
{
    double lossV = 0.0L;
    int i;

    for (i = 0; i < trainDataSet.size(); i++) {
        lossV -= trainDataSet[i].label * log(sigmoidCalc(wxbCalc(trainDataSet[i])));
        lossV -= (1 - trainDataSet[i].label) * log(1 - sigmoidCalc(wxbCalc(trainDataSet[i])));
    }
    lossV /= trainDataSet.size();
    return lossV;
}


double LR::gradientSlope(const vector<Data> &dataSet, int index, const vector<double> &sigmoidVec, 
	const vector<int>& indexes)
{
    double gsV = 0.0L;
    int i,k;
    double sigv, label;
   

    for (i = 0; i < indexes.size(); i++) {
        sigv = sigmoidVec[i];
		k = indexes[i];
        label = dataSet[k].label;
        gsV += (sigv-label ) * (dataSet[k].features[index]);
    }

    gsV = gsV / indexes.size();   // 归一化的除法
    return gsV;
}

void LR::train()
{
    double sigmoidVal;
    double wxbVal,loss;
    int i, j;
    auto start1 = chrono::steady_clock::now();
	vector<int> indexes;
	indexes.reserve(trainDataSet.size() * batch);
	for (int i = 0; i < trainDataSet.size() * batch; ++i) {
		indexes.push_back(i);
	}

    for (i = 0; i < maxIterTimes; i++) {
        vector<double> sigmoidVec;
        sigmoidVec.reserve(Num);
        
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		shuffle(indexes.begin(), indexes.end(), default_random_engine(seed));

		for (int j : indexes) {
            wxbVal = wxbCalc(trainDataSet[j]);
            sigmoidVal = sigmoidCalc(wxbVal);
            sigmoidVec.push_back(sigmoidVal);
        }

        for (j = 0; j < param.wtSet.size(); j++) {
            param.wtSet[j] -= stepSize * (gradientSlope(trainDataSet, j, sigmoidVec,indexes)+lambda* param.wtSet[j]);
        }
        if (LOSS&& (i%2==0)){
         loss=lossCal();
         acurracy();
         cout<<"训练损失."<<loss<< endl;
        }
        

    }
    auto end1 = chrono::steady_clock::now();
    chrono::duration<double> elapsed1 = end1 - start1;
    cout<< "训练时间time: "  << elapsed1.count() << "s" << endl;
}

void LR::acurracy()
{
    double tanv,sigv;
    int predictVal;
    int Count = 0;
    double accurate = 0;
  
    for (int j = 0; j < trainDataSet.size(); j++) {       //有待优化
           sigv = sigmoidCalc(wxbCalc(trainDataSet[j]));
           predictVal = (sigv >= predictTrueThresh) ? 1 : 0;
            if (predictVal ==trainDataSet[j].label) {
                  Count++;
            }         
     }
    accurate = ((double)Count) / trainDataSet.size();
    cout << "the train accuracy is " << accurate << endl;
}
void LR::predict()
{
    double sigVal;
    int predictVal;
    predictVec.reserve(Num);

    loadTestData();
    normalset(testDataSet);
    for (int j = 0; j < testDataSet.size(); j++) {
        sigVal = sigmoidCalc(wxbCalc(testDataSet[j]));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        predictVec.push_back(predictVal);
    }

    storePredict(predictVec);
}

int LR::loadModel()
{
    string line;
    int i;
    vector<double> wtTmp;
    double dbt;

    ifstream fin(weightParamFile.c_str());
    if (!fin) {
        cout << "打开模型参数文件失败" << endl;
        exit(0);
    }

    getline(fin, line);
    stringstream sin(line);
    for (i = 0; i < featuresNum; i++) {
        char c = sin.peek();
        if (c == -1) {
            cout << "模型参数数量少于特征数量，退出." << endl;
            return -1;
        }
        sin >> dbt;
        wtTmp.push_back(dbt);
    }
    param.wtSet.swap(wtTmp);
    fin.close();
    return 0;
}

int LR::storeModel()
{
    string line;
    int i;

    ofstream fout(weightParamFile.c_str());
    if (!fout.is_open()) {
        cout << "打开模型参数文件失败" << endl;
    }
    if (param.wtSet.size() < featuresNum) {
        cout << "wtSet size is " << param.wtSet.size() << endl;
    }
    for (i = 0; i < featuresNum; i++) {
        fout << param.wtSet[i] << " ";
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
        feature.reserve(1001);
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
                    cout << "测试文件数据格式不正确." << endl;
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
    awFile.reserve(Num);

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
    auto start = chrono::steady_clock::now();
      int mypid = getpid();
       cpu_set_t mask;
     int aff =sched_getaffinity( mypid, sizeof(mask), &mask) ;
    cout<<aff;
    CPU_ZERO( &mask );
  
    CPU_SET(0, &mask );
    if( sched_setaffinity( mypid, sizeof(mask), &mask ) == -1 ){
         printf("WARNING: Could not set CPU Affinity, continuing...\n");
    }
     sched_setaffinity( mypid, sizeof(mask), &mask );
     cout<<sched_getaffinity( mypid, sizeof(mask), &mask) ;
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
    logist.train();

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

auto end = chrono::steady_clock::now();
chrono::duration<double> elapsed = end - start;
cout<< "运行时间time: "  << elapsed.count() << "s" << endl;

    return 0;
}
