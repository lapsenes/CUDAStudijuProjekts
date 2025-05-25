#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "dense_layer.h"

#define INPUT_DIM 4
#define HIDDEN_DIM 6
#define OUTPUT_DIM 3
#define EPOCHS 600
#define LR 0.1f
#define TRAIN_RATIO 0.8f
#define SCALE_FACTOR 1.0f // Scaling factor for input data

// value mapping for labels
std::unordered_map<std::string, int> label_map = {
    {"Iris-setosa", 0},
    {"Iris-versicolor", 1},
    {"Iris-virginica", 2}
};

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

bool read_csv_data(const std::string& filename, std::vector<float>& X, std::vector<int>& y) { 
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        int col = 0;
        std::vector<float> sample(INPUT_DIM); 
        std::string label;
        while (std::getline(ss, val, ',')) {
            val = trim(val);
            if (col < INPUT_DIM) {
                try {
                    sample[col] = std::stof(val);
                } catch (...) {
                    sample[col] = 0;
                }
            } else {
                label = val; 
            }
            col++;
        }
        if (col == INPUT_DIM + 1 && label_map.count(label)) { 
            X.insert(X.end(), sample.begin(), sample.end());
            y.push_back(label_map[label]);
        }
    }
    return true;
}

// main
int main() {
    std::vector<float> full_X;
    std::vector<int> full_y;
    
    // Timing the data loading
    auto data_load_start = std::chrono::high_resolution_clock::now();
    if (!read_csv_data("iris.csv", full_X, full_y)) { 
        std::cerr << "Failed to load iris.csv\n";
        return 1;
    }
    auto data_load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> data_load_time = data_load_end - data_load_start;
    std::cout << "Data Loading Time: " << data_load_time.count() << "s\n";

    int total_samples = full_y.size();
    int train_samples = static_cast<int>(total_samples * TRAIN_RATIO);
    int test_samples = total_samples - train_samples;

    std::vector<std::pair<std::vector<float>, int>> dataset(total_samples);
    for (int i = 0; i < total_samples; ++i) {
        dataset[i].first = std::vector<float>(full_X.begin() + i * INPUT_DIM, full_X.begin() + (i + 1) * INPUT_DIM); 
        dataset[i].second = full_y[i]; 
    }

    std::srand(static_cast<unsigned>(time(nullptr)));

    // dataset shuffling
    for (int i = dataset.size() - 1; i > 0; --i) {
        int j = std::rand() % (i + 1);
        std::swap(dataset[i], dataset[j]);
    }

    std::vector<float> X_train(train_samples * INPUT_DIM); 
    std::vector<int> y_train(train_samples);
    std::vector<float> X_test(test_samples * INPUT_DIM);
    std::vector<int> y_test(test_samples);

    // populate training and testing data
    for (int i = 0; i < train_samples; ++i) { 
        for (int j = 0; j < INPUT_DIM; ++j) 
            X_train[i * INPUT_DIM + j] = dataset[i].first[j];
        y_train[i] = dataset[i].second;
    }

    for (int i = 0; i < test_samples; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j)
            X_test[i * INPUT_DIM + j] = dataset[train_samples + i].first[j];
        y_test[i] = dataset[train_samples + i].second;
    }

    // Timing the normalization
    auto normalization_start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < INPUT_DIM; ++j) {
        float sum = 0, sum_sq = 0;
        for (int i = 0; i < train_samples; ++i) {
            float val = X_train[i * INPUT_DIM + j];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / train_samples;
        float std = std::sqrt(sum_sq / train_samples - mean * mean + 1e-8f);

        // scaling the data
        for (int i = 0; i < train_samples; ++i)
            X_train[i * INPUT_DIM + j] = (X_train[i * INPUT_DIM + j] - mean) / std * SCALE_FACTOR;
        for (int i = 0; i < test_samples; ++i)
            X_test[i * INPUT_DIM + j] = (X_test[i * INPUT_DIM + j] - mean) / std * SCALE_FACTOR;
    }
    auto normalization_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> normalization_time = normalization_end - normalization_start;
    std::cout << "Normalization Time: " << normalization_time.count() << "s\n";

    float *W1, *b1, *W2, *b2;
    cudaMallocManaged(&W1, INPUT_DIM * HIDDEN_DIM * sizeof(float));
    cudaMallocManaged(&b1, HIDDEN_DIM * sizeof(float));
    cudaMallocManaged(&W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    cudaMallocManaged(&b2, OUTPUT_DIM * sizeof(float));

    std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; ++i) W1[i] = dist(rng);
    for (int i = 0; i < HIDDEN_DIM; ++i) b1[i] = 0.0f;
    for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; ++i) W2[i] = dist(rng);
    for (int i = 0; i < OUTPUT_DIM; ++i) b2[i] = 0.0f;

    // Memory allocation
    float *X, *hidden, *logits, *probs, *dY;
    cudaMallocManaged(&X, train_samples * INPUT_DIM * sizeof(float));
    cudaMallocManaged(&hidden, train_samples * HIDDEN_DIM * sizeof(float));
    cudaMallocManaged(&logits, train_samples * OUTPUT_DIM * sizeof(float));
    cudaMallocManaged(&probs, train_samples * OUTPUT_DIM * sizeof(float));
    cudaMallocManaged(&dY, train_samples * OUTPUT_DIM * sizeof(float));
    std::copy(X_train.begin(), X_train.end(), X);

    int* d_y_train;
    cudaMallocManaged(&d_y_train, train_samples * sizeof(int));
    std::copy(y_train.begin(), y_train.end(), d_y_train);

    int* d_y_test;
    cudaMallocManaged(&d_y_test, test_samples * sizeof(int));
    std::copy(y_test.begin(), y_test.end(), d_y_test);


    // Memory allocation for gradients
    float *dW1, *db1, *dW2, *db2;
    cudaMallocManaged(&dW1, INPUT_DIM * HIDDEN_DIM * sizeof(float));
    cudaMallocManaged(&db1, HIDDEN_DIM * sizeof(float));
    cudaMallocManaged(&dW2, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    cudaMallocManaged(&db2, OUTPUT_DIM * sizeof(float));

    // Loss and accuracy reporting
    float lambda = 1e-4f;
    float clip = 1.0f;

    float* loss_array;
    cudaMallocManaged(&loss_array, train_samples * sizeof(float));
    int* correct_array;
    cudaMallocManaged(&correct_array, train_samples * sizeof(int));

    // Time for training
    auto training_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        dense_forward(X, W1, b1, hidden, train_samples, INPUT_DIM, HIDDEN_DIM);
        relu_forward(hidden, train_samples * HIDDEN_DIM);
        dense_forward(hidden, W2, b2, logits, train_samples, HIDDEN_DIM, OUTPUT_DIM);
        softmax_forward(logits, probs, train_samples, OUTPUT_DIM);

        compute_loss_and_gradient_cuda(probs, d_y_train, dY, loss_array, train_samples, OUTPUT_DIM);

        float loss = 0.0f;
        for (int i = 0; i < train_samples; ++i) loss += loss_array[i];
        loss /= train_samples;

        dense_backward(dY, hidden, dW2, db2, train_samples, HIDDEN_DIM, OUTPUT_DIM);
        dense_backward(dY, X, dW1, db1, train_samples, INPUT_DIM, HIDDEN_DIM);

        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; ++i) W1[i] -= LR * (fminf(fmaxf(dW1[i], -clip), clip) + lambda * W1[i]);
        for (int i = 0; i < HIDDEN_DIM; ++i) b1[i] -= LR * fminf(fmaxf(db1[i], -clip), clip);
        for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; ++i) W2[i] -= LR * (fminf(fmaxf(dW2[i], -clip), clip) + lambda * W2[i]);
        for (int i = 0; i < OUTPUT_DIM; ++i) b2[i] -= LR * fminf(fmaxf(db2[i], -clip), clip);

        if (epoch % 100 == 0 || epoch == EPOCHS - 1) {
            compute_accuracy_cuda(probs, d_y_train, correct_array, train_samples, OUTPUT_DIM);
            int correct = 0;
            for (int i = 0; i < train_samples; ++i) correct += correct_array[i];
            float acc = static_cast<float>(correct) / train_samples;
            std::cout << "Epoch " << epoch << " - Loss: " << loss << ", Accuracy: " << acc << std::endl;
        }
    }

    auto training_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_time = training_end - training_start;
    std::cout << "Training Time: " << training_time.count() << "s\n";

    // Time for testing
    auto testing_start = std::chrono::high_resolution_clock::now();

    float* Xtest_dev, *Htest, *Ltest, *SoftTest;
    cudaMallocManaged(&Xtest_dev, test_samples * INPUT_DIM * sizeof(float));
    cudaMallocManaged(&Htest, test_samples * HIDDEN_DIM * sizeof(float));
    cudaMallocManaged(&Ltest, test_samples * OUTPUT_DIM * sizeof(float));
    cudaMallocManaged(&SoftTest, test_samples * OUTPUT_DIM * sizeof(float));
    std::copy(X_test.begin(), X_test.end(), Xtest_dev);
    dense_forward(Xtest_dev, W1, b1, Htest, test_samples, INPUT_DIM, HIDDEN_DIM);
    relu_forward(Htest, test_samples * HIDDEN_DIM);
    dense_forward(Htest, W2, b2, Ltest, test_samples, HIDDEN_DIM, OUTPUT_DIM);
    softmax_forward(Ltest, SoftTest, test_samples, OUTPUT_DIM);

    int* test_correct_array;
    cudaMallocManaged(&test_correct_array, test_samples * sizeof(int));
    compute_accuracy_cuda(SoftTest, d_y_test, test_correct_array, test_samples, OUTPUT_DIM);

    int test_correct = 0;
    for (int i = 0; i < test_samples; ++i) test_correct += test_correct_array[i];
    float test_acc = static_cast<float>(test_correct) / test_samples;

    auto testing_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> testing_time = testing_end - testing_start;
    std::cout << "Testing Time: " << testing_time.count() << "s\n";

    std::cout << "Test Accuracy: " << test_acc << std::endl;

    // Writing report
    std::ofstream report("training_report.txt");
    report << "Data Loading Time: " << data_load_time.count() << "s\n";
    report << "Normalization Time: " << normalization_time.count() << "s\n";
    report << "Training Time: " << training_time.count() << "s\n";
    report << "Testing Time: " << testing_time.count() << "s\n";
    report << "Test Accuracy: " << test_acc << std::endl;
    report.close();

    // Clean up memory
    cudaFree(loss_array);
    cudaFree(correct_array);
    cudaFree(test_correct_array);
    cudaFree(d_y_train);
    cudaFree(d_y_test);

    return 0;
}
