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
#define Num  200001

struct Data {
    vector<double> features;
    int label;
    Data(vector<double> f, int l) : features(f), label(l)
    {}
};
struct Param {
    vector<double> w1;
    double w2;
    double b1;
    double b2;
};
struct Gradient{
    vector<double> dw1;
    //double dw1;
    double dw2;
    double db1;
    double db2;
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
    vector<float>sum;
    vector<float>sd;
	Param param;
    Gradient gd;
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
    double tanhCalc(const double& wxb);
	double lossCal();
    bool normalset(vector<Data>& dataset,vector<float> & sum,vector<float> & sd);
    bool normaltest(vector<Data>& dataset,vector<float> & sum,vector<float> & sd);
	double gradientSlope(const vector<Data>& dataSet, const vector<double>& a1,
		const vector<double>& a2,const int k);
    
private:
	int featuresNum;
	const double wtInitV = 0.3;     //参数取消
    const double lambda =0.11;
	const double stepSize = 0.5;
	const int maxIterTimes = 6;
	const double predictTrueThresh = 0.395;
	const int train_show_step = 1000;
	const double batch = 256;
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
    int num=0;

    while (infile&&(num<3500)) {
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
            num++;
        }
    }
    infile.close();
    auto end2 = chrono::steady_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    cout<< "读取数据time:. "  << elapsed2.count() << "s" << endl;
    return true;
}

bool LR::normalset(vector<Data>& dataset,vector<float>& sum,vector<float>& sd) {
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

bool LR::normaltest(vector<Data>& dataset,vector<float>& sum,vector<float>& sd) {
	int dim = dataset[0].features.size();
	int num = dataset.size();
	float sumtem = 0, sdtem = 0;
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
    for (i = 0; i < 1000; i++) {
        k = rand() % ( 100) / (float)(101);
        param.w1.push_back(k*0.01);
        gd.dw1.push_back(0);
    }
      k = rand() % ( 100) / (float)(101);
      param.w2=k*0.01;
      k = rand() % ( 100) / (float)(101);
      param.b1=k*0.01;
      k = rand() % ( 100) / (float)(101);
      param.b2=k*0.01;
}

bool LR::init()
{
    trainDataSet.clear();
    trainDataSet.reserve(Num);
    bool status = loadTrainData();
    if (status != true) {
        return false;
    }
    normalset( trainDataSet,sum,sd);
    featuresNum = trainDataSet[0].features.size();
    param.w1.clear();
    param.w1.reserve(1001);
    initParam();
    return true;
}


double LR::wxbCalc(const Data &data)
{
    double mulSum = 0.0L;
    int i;
    double wtv, feav;
    for (i = 0; i < param.w1.size(); i++) {
        wtv = param.w1[i];
        feav = data.features[i];
        mulSum += wtv * feav;
    }
    mulSum+= param.b1;
    return mulSum;
}

inline double LR::sigmoidCalc(const double &wxb)
{
    double expv = exp(-1 * wxb);
    double expvInv = 1 / (1 + expv);
    return expvInv;
}
inline double LR::tanhCalc(const double &wxb)
{
    double expv = exp(-1 * wxb);
    double expv0 = exp(1 * wxb);
    double expvInv =(expv0-expv)/ (expv0+expv);
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


double LR::gradientSlope(const vector<Data> &dataSet, const vector<double> & a1, 
	const vector<double> & a2,const int k)
{
    vector<double> dz2;
    vector<double> dz1;
    double gsV = 0.0L;
    int i,num,nbatch;
    num =dataSet.size()/batch;
    if(k==num){
        num=dataSet.size();
        nbatch= dataSet.size()-k*batch;
    }
    else{
        num=(k+1)*batch;
        nbatch=batch;
    }
    double dz2v = 0;
	double dz1v = 0;
	double dw2 = 0;
	double db2 = 0;
	double db1 = 0;
    vector<double> dw1;
    for(int n=0;n<dataSet[0].features.size();n++)
    {
        dw1.push_back(0);
    }
    
    for (i = k*batch; i < num; i++) {
        dz2v =a2[i]- dataSet[i].label;             
        dw2 +=dz2v*a1[i];
        db2  +=dz2v;
        dz1v =param.w2*dz2v*(1-a1[i]*a1[i]);
        db1 +=dz1v;
        for(int n=0;n<dataSet[0].features.size();n++){
             dw1[n] +=dz1v* (dataSet[i].features[n]);
        }
    }
     for(int n=0;n<dataSet[0].features.size();n++)
    {
        gd.dw1[n ]= (dw1[n]/nbatch)+(param.w1[n])*lambda; 
    }

   // 归一化的除法
    db1 = db1/nbatch;
    dw2 = (dw2/nbatch)+(param.w2*lambda); 
    db2 = db2/nbatch;
    gd.dw2=dw2;
    gd.db1=db1;
    gd.db2=db2;
    return 0;
}

void LR::train()
{
    double z1,z2;
    double wxbVal,loss;
    int i, j,num_batch;
    auto start1 = chrono::steady_clock::now();
	vector<int> indexes;
	indexes.reserve(trainDataSet.size() * batch);
	for (int i = 0; i < trainDataSet.size() * batch; ++i) {
		indexes.push_back(i);
	}
    num_batch =trainDataSet.size()/batch;
    for (i = 0; i < maxIterTimes; i++) {
        vector<double> sigmoidVec;
        vector<double> a1;
        vector<double> a2;
        sigmoidVec.reserve(Num);
        a1.reserve(Num);
        a2.reserve(Num);
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		shuffle(indexes.begin(), indexes.end(), default_random_engine(seed));
        
        for(int k=0;k<num_batch;k++){
            //前向传播
            for (int j =k*batch;j<(k+1)*batch;j++) {
            wxbVal = wxbCalc(trainDataSet[j])+param.b1;
            z1 = tanhCalc(wxbVal);
            a1.push_back(z1);
            }
           for (int j =k*batch;j<(k+1)*batch;j++) {
            wxbVal = a1[j]*param.w2+param.b2;
            z2 = sigmoidCalc(wxbVal);
            a2.push_back(z2);
            }
            //反向传播
            gradientSlope(trainDataSet,a1,a2,k);
            for (j = 0; j < param.w1.size(); j++) {
            param.w1[j] -= stepSize * gd.dw1[j];
            }
            param.w2 -= stepSize * gd.dw2;
            param.b1 -= stepSize * gd.db1;
            param.b2 -= stepSize * gd.db2;
        }
        int subnum=0;
        subnum=trainDataSet.size()-num_batch*batch;
        for (int j =num_batch*batch;j<trainDataSet.size();j++) {
            wxbVal = wxbCalc(trainDataSet[j])+param.b1;
            z1 = tanhCalc(wxbVal);
            a1.push_back(z1);
         }
        for (int j =num_batch*batch;j<trainDataSet.size();j++) {
            wxbVal = a1[j]*param.w2+param.b2;
            z2 = sigmoidCalc(wxbVal);
            a2.push_back(z2);
         }
         //反向传播
         gradientSlope(trainDataSet, a1,a2,num_batch);
         
        for (j = 0; j < param.w1.size(); j++) {
            param.w1[j] -= stepSize * gd.dw1[j];
         }
            param.w2 -= stepSize * gd.dw2;
            param.b1 -= stepSize * gd.db1;
            param.b2 -= stepSize * gd.db2;
    
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

void LR::predict()
{
    double tanv,sigv;
    double z1v,z2v;
    int predictVal;
    predictVec.reserve(Num);
    loadTestData();
    normaltest(testDataSet,sum,sd);
    vector<double> a1;

    for (int j = 0; j < testDataSet.size(); j++) {
            tanv= wxbCalc(testDataSet[j])+param.b1;
            z1v = tanhCalc(tanv);
            a1.push_back(z1v);
            }
    for (int j = 0; j < testDataSet.size(); j++) {
            sigv = a1[j]*param.w2+param.b2;
            z2v = sigmoidCalc(sigv);
            predictVal = z2v >= predictTrueThresh ? 1 : 0;
            predictVec.push_back(predictVal);
     }

    storePredict(predictVec);
}

void LR::acurracy()
{
    double tanv,sigv;
    double z1v,z2v;
    int predictVal;
  
    int Count = 0;
    double accurate = 0;
    
    vector<double> a1;

    for (int j = 0; j < trainDataSet.size(); j++) {
            tanv= wxbCalc(trainDataSet[j])+param.b1;
            z1v = tanhCalc(tanv);
            a1.push_back(z1v);
            }
    for (int j = 0; j < trainDataSet.size(); j++) {       //有待优化
            sigv = a1[j]*param.w2+param.b2;
            z2v = sigmoidCalc(sigv);
            predictVal = z2v >= predictTrueThresh ? 1 : 0;
            if (predictVal ==trainDataSet[j].label) {
                  Count++;
            }         
     }
    accurate = ((double)Count) / trainDataSet.size();
    cout << "the train accuracy is " << accurate << endl;
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
    param.w1.swap(wtTmp);
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
    if (param.w1.size() < featuresNum) {
        cout << "wtSet size is " << param.w1.size() << endl;
    }
    for (i = 0; i < featuresNum; i++) {
        fout << param.w1[i] << " ";
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
    //logist.storeModel();

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
