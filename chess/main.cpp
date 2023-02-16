#include <Windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <math.h>
#include "../Eigen/Dense"
#include "util.h"
#include "Analyze.h"
#include "neural.h"

using namespace std;
using namespace Eigen;
using namespace util;

int main()
{
    srand((unsigned int)time(0));

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

    vector<int> param = {insz, 100, outsz};
    Neural network;
    network.InitBuild(param);

    // 迭代次数
    int epochs = 1;

    // 批处理大小
    int batch = 64;

    MatrixXf mi(insz, batch);
    MatrixXf mt(outsz, batch);
    
    MatrixXf so;
    float loss = 0;
    test.ReadRandom(mi, mt, batch);
    network.CompareSample(mi, mt, so, loss);
    cout << "before:" << loss << endl;

    for (int i = 0; i < epochs; i++) {
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
    }

    test.ReadRandom(mi, mt, batch);
    network.CompareSample(mi, mt, so, loss);
    cout << "after:" << loss << endl;

    return 0;
}