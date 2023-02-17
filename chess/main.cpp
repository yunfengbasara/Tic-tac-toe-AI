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
#include "../Eigen/Core"

using namespace std;
using namespace std::chrono;
using namespace Eigen;
using namespace util;

int main()
{
    srand((unsigned int)time(0));

    // ����������ؼ�¼
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

    // ��0��ʼ������
    network.InitBuild({ insz, 100, outsz });

    // ����һ��ѵ��������翪ʼ
    //network.Load(record);

    // ��������
    int epochs = 1;

    // �������С
    int batch = 64;

    MatrixXf mi(insz, batch);
    MatrixXf mt(outsz, batch);
    
    MatrixXf so;
    float loss = 0;
    test.ReadRandom(mi, mt, batch);
    network.CompareSample(mi, mt, so, loss);
    cout << "before:" << loss << endl;

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

        auto elapse = steady_clock::now() - start;
        auto sec = duration_cast<seconds>(elapse);
        cout << "use " << sec.count() << " seconds" << endl;
    }

    // ��¼ѵ���������
    network.Save(record);

    test.ReadRandom(mi, mt, batch);
    network.CompareSample(mi, mt, so, loss);
    cout << "after:" << loss << endl;

    return 0;
}