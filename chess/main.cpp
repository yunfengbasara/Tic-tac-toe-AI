#include <Windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <chrono>
#include <math.h>
#include "util.h"
#include "Analyze.h"
#include "neural.h"
#include "neuralex.h"
#include "../Eigen/Core"

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace util;

// 从本地训练记录开始
#define START_FROM_RECORD

int main()
{
    HOSTMatrix ma(1130, 1230);
    ma.setRandom();

    NeuralEx mu;

    cout << "finsh";
    return 0;

    srand((unsigned int)time(0));

    // 网络参数本地记录
    std::wstring record = GetCurrentDir() + L"neural";

    std::wstring ti = GetCurrentDir() + L"t10k-images.idx3-ubyte";
    std::wstring tt = GetCurrentDir() + L"t10k-labels.idx1-ubyte";

    Analyze_IDX test;
    test.SetSample(ti, tt);

    std::wstring si = GetCurrentDir() + L"train-images.idx3-ubyte";
    std::wstring st = GetCurrentDir() + L"train-labels.idx1-ubyte";

    Analyze_IDX train;
    train.SetSample(si, st);

    int insz, outsz;
    train.GetInOut(insz, outsz);

    Neural network;

#ifdef START_FROM_RECORD
    network.Load(record);
#else
    network.InitBuild({ insz, 100, outsz });
#endif

    network.SetLearnRate(0.3);

    // 迭代次数
    int epochs = 1;

    // 批处理大小
    int batch = 64;

    MatrixXf mi(insz, batch);
    MatrixXf mt(outsz, batch);
    
    MatrixXf so;
    float loss = 0;

    for (int i = 0; i < epochs; i++) {
        // 计算loss
        test.ReadRandom(mi, mt, batch);
        network.CompareSample(mi, mt, so, loss);
        std::cout << "loss:" << loss << endl;

        auto start = steady_clock::now();

        train.Shuffle();

        int bat = batch;
        while (bat == batch) {
            bat = train.ReadSample(mi, mt, batch);
            if (bat == 0) {
                break;
            }

            if (!network.SetSample(mi, mt)) {
                break;
            }

            network.SGD();
        }

        // 计算loss
        test.ReadRandom(mi, mt, batch);
        network.CompareSample(mi, mt, so, loss);
        std::cout << "loss:" << loss << endl;

        // 运行时间
        auto elapse = steady_clock::now() - start;
        auto sec = duration_cast<seconds>(elapse);
        std::cout << "use " << sec.count() << " seconds" << endl;
    }

    // 记录训练后的网络
    //network.Save(record);

    // 计算正确率
    int cnt = 0;
    for (int i = 0; i < batch; i++) {
        const auto& col1 = mt.col(i);
        const auto& col2 = so.col(i);
        int idx1 = 0, idx2 = 0;
        for (int j = 0; j < col1.rows(); j++) {
            if (col1(j) > col1(idx1)) {
                idx1 = j;
            }

            if (col2(j) > col2(idx2)) {
                idx2 = j;
            }
        }
        if (idx1 == idx2) {
            cnt++;
        }
    }

    std::cout << (float)cnt / batch << endl;

    return 0;
}