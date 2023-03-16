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
#include "type.h"
#include "../Eigen/Core"
#include "TicRule.h"

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace util;
using namespace chess;

void checktest(MatrixXf mt, MatrixXf so) {
    int batch = mt.cols();

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

    std::cout << (float)cnt / batch * 100 << "%" << endl;
}

// 从本地训练记录开始
//#define START_FROM_RECORD

// cuda加速
#define CUDA_NEURAL

int main() {
    auto seed = std::default_random_engine(std::random_device()());

    // 随机棋盘记录
    typedef std::pair<Matrix3i, Matrix3f> SAMPLE;
    std::vector<SAMPLE> record;

    // 随机产生
    int count = 3200;
    record.resize(count);

    // 落子范围随机走法
    std::vector<int> steps = { 0,1,2,3,4,5,6,7,8 };

    // 规则
    Tic ticRule;
    // 结果
    Tic::GameType result;
    // 验证最后一步
    int last;

    for (auto& item : record) {
        std::shuffle(steps.begin(), steps.end(), seed);
        ticRule.Reset();

        if (!ticRule.Create(steps, item.first, result, last)) {
            return -1;
        }

        if (result == Tic::GameType::DRAW) {
            item.second.setConstant(0.1f);
            continue;
        }

        // 去掉最后一步
        item.first(last / 3, last % 3) = 0;

        item.second.setConstant(0);
        item.second(last / 3, last % 3) = 1.0f;
    }

    // 训练
    int epochs = 100;
    int batch = 64;

    MatrixXf mi(9, batch);
    MatrixXf mt(9, batch);
    MatrixXf so;
    float loss = 0;

    srand((unsigned int)time(0));
    NeuralEx network;
    network.SetCostType(NeuralEx::CrossEntropy);
    network.InitBuild({ 9, 100, 9 });
    network.SetLearnRate(0.1);
    network.SetRegularization(5.0);
    network.SetTotalItem(count);

    for (int i = 0; i < epochs; i++) {
        std::shuffle(record.begin(), record.end(), seed);

        for (int k = 0; k < count; k += batch) {
            int sz = batch;
            if (k + batch > count) {
                sz = count - k;
            }

            if (sz == 0) {
                std::cout << "batch = 0" << std::endl;
                continue;
            }

            if (mi.cols() != sz) {
                mi.resize(NoChange, sz);
                mt.resize(NoChange, sz);
            }

            for (int m = 0; m < sz; m++) {
                const SAMPLE& s = record[k + m];
                mi.col(m) = Map<VectorXi>((int*)s.first.data(), 9).cast<float>();
                mt.col(m) = Map<VectorXf>((float*)s.second.data(), 9);
            }

            if (!network.SetSample(mi, mt)) {
                break;
            }

            network.SGD();
        }
    }

    // 验证
    network.CompareSample(mi, mt, so, loss);
    std::cout << "loss " << loss <<  std::endl;

    checktest(mt, so);

    //for (int i = 0; i < mi.cols(); i++) {
    //    Matrix3f board = Map<Matrix3f>((float*)mi.col(i).data());
    //    Matrix3f target = Map<Matrix3f>((float*)mt.col(i).data());
    //    Matrix3f answer = Map<Matrix3f>((float*)so.col(i).data());

    //    std::cout << "board" << std::endl;
    //    std::cout << board << std::endl;
    //    std::cout << "target" << std::endl;
    //    std::cout << target << std::endl;
    //    std::cout << "AI answer" << std::endl;
    //    std::cout << answer << std::endl;
    //}
    
    return 0;
}

int mainHandWrite()
{
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
    int items = train.GetTotalItems();

#ifdef CUDA_NEURAL
    NeuralEx network;
    network.SetCostType(NeuralEx::CrossEntropy);
#else 
    Neural network;
#endif

#ifdef START_FROM_RECORD
    network.Load(record);
#else
    network.InitBuild({ insz, 160, outsz });
#endif

    network.SetLearnRate(0.1);
    network.SetRegularization(5.0);
    network.SetTotalItem(items);

    // 迭代次数
    int epochs = 1;

    // 批处理大小
    int batch = 64;

    // 训练前计算正确率
    MatrixXf tmi(insz, batch);
    MatrixXf tmt(outsz, batch);
    MatrixXf tso;
    float tloss = 0;
    test.ReadRandom(tmi, tmt, batch);

    network.CompareSample(tmi, tmt, tso, tloss);
    std::cout << "before train loss:" << tloss << endl;
    checktest(tmt, tso);

    // 训练数据
    MatrixXf mi(insz, batch);
    MatrixXf mt(outsz, batch);

    MatrixXf so;
    float loss = 0;

    for (int i = 0; i < epochs; i++) {
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

        // 测试训练数据
        train.ReadRandom(mi, mt, batch);
        network.CompareSample(mi, mt, so, loss);

        // 运行时间
        auto elapse = steady_clock::now() - start;
        auto msec = duration_cast<milliseconds>(elapse);
        std::cout << 
            " epoch " << i + 1 << 
            " loss " << loss <<
            " use " << msec.count() << " milliseconds" << endl;
    }

    // 训练后计算正确率(同一组数据)
    network.CompareSample(tmi, tmt, tso, tloss);
    std::cout << "after train loss:" << tloss << endl;
    checktest(tmt, tso);

    std::cout << "do you want to save the nerual network? Y/N" << endl;

    char keyin;
    std::cin >> keyin;
    if (keyin == 'y' || keyin == 'Y') {
        network.Save(record);
    }

    return 0;
}