// cpu_main.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <algorithm>

// Network configuration
#define INPUT_DIM 4      // Each Iris sample has 4 features
#define HIDDEN_DIM 6     // Hidden layer size
#define OUTPUT_DIM 3     // Three possible classes
#define EPOCHS 600       // Number of training epochs
#define LR 0.1f          // Learning rate
#define TRAIN_RATIO 0.8f // 80% for training, 20% for testing

// Map from string class names to integer labels
std::unordered_map<std::string, int> label_map = {
    {"Iris-setosa", 0},
    {"Iris-versicolor", 1},
    {"Iris-virginica", 2}
};

// Removes leading and trailing whitespace
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

// Loads and parses CSV into feature matrix X and label vector y
bool read_csv_data(const std::string& filename, std::vector<float>& X, std::vector<int>& y) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;

    // Skip header if present
    std::getline(file, line);

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
                    sample[col] = 0.0f; // Fallback to 0 on parse error
                }
            } else {
                label = val; // Last column is label
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

// Matrix multiplication with optional bias addition
void matmul(const float* A, const float* B, float* C, const float* bias, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// Applies ReLU activation in-place
void relu(float* A, int size) {
    for (int i = 0; i < size; ++i)
        A[i] = std::max(0.0f, A[i]);
}

// Applies softmax row-wise for a 2D matrix
void softmax(float* A, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float max_val = A[i * cols];
        for (int j = 1; j < cols; ++j)
            max_val = std::max(max_val, A[i * cols + j]);

        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            A[i * cols + j] = expf(A[i * cols + j] - max_val);
            sum += A[i * cols + j];
        }
        for (int j = 0; j < cols; ++j)
            A[i * cols + j] /= sum;
    }
}

// Computes categorical cross-entropy loss
float cross_entropy_loss(float* probs, const std::vector<int>& labels, int rows, int cols) {
    float loss = 0.0f, eps = 1e-8;
    for (int i = 0; i < rows; ++i)
        loss += -logf(probs[i * cols + labels[i]] + eps);
    return loss / rows;
}

// Computes accuracy by comparing argmax(predictions) to labels
float compute_accuracy(float* probs, const std::vector<int>& labels, int rows) {
    int correct = 0;
    for (int i = 0; i < rows; ++i) {
        int max_idx = 0;
        for (int j = 1; j < OUTPUT_DIM; ++j)
            if (probs[i * OUTPUT_DIM + j] > probs[i * OUTPUT_DIM + max_idx])
                max_idx = j;
        if (max_idx == labels[i]) ++correct;
    }
    return static_cast<float>(correct) / rows;
}

int main() {
    // Load dataset
    std::vector<float> X, X_train, X_test;
    std::vector<int> y, y_train, y_test;
    if (!read_csv_data("iris.csv", X, y)) return 1;

    int total_samples = y.size();
    int train_samples = static_cast<int>(total_samples * TRAIN_RATIO);
    int test_samples = total_samples - train_samples;

    // Bundle features and labels together for shuffling
    std::vector<std::pair<std::vector<float>, int>> dataset(total_samples);
    for (int i = 0; i < total_samples; ++i) {
        dataset[i].first = std::vector<float>(X.begin() + i * INPUT_DIM, X.begin() + (i + 1) * INPUT_DIM);
        dataset[i].second = y[i];
    }

    std::shuffle(dataset.begin(), dataset.end(), std::default_random_engine(42));

    // Split into training and test sets
    X_train.resize(train_samples * INPUT_DIM);
    y_train.resize(train_samples);
    X_test.resize(test_samples * INPUT_DIM);
    y_test.resize(test_samples);

    for (int i = 0; i < train_samples; ++i) {
        std::copy(dataset[i].first.begin(), dataset[i].first.end(), X_train.begin() + i * INPUT_DIM);
        y_train[i] = dataset[i].second;
    }
    for (int i = 0; i < test_samples; ++i) {
        std::copy(dataset[train_samples + i].first.begin(), dataset[train_samples + i].first.end(), X_test.begin() + i * INPUT_DIM);
        y_test[i] = dataset[train_samples + i].second;
    }

    // Normalize features using training mean and std
    for (int j = 0; j < INPUT_DIM; ++j) {
        float sum = 0, sq_sum = 0;
        for (int i = 0; i < train_samples; ++i) {
            float v = X_train[i * INPUT_DIM + j];
            sum += v;
            sq_sum += v * v;
        }
        float mean = sum / train_samples;
        float std = std::sqrt(sq_sum / train_samples - mean * mean + 1e-8f);
        for (int i = 0; i < train_samples; ++i)
            X_train[i * INPUT_DIM + j] = (X_train[i * INPUT_DIM + j] - mean) / std;
        for (int i = 0; i < test_samples; ++i)
            X_test[i * INPUT_DIM + j] = (X_test[i * INPUT_DIM + j] - mean) / std;
    }

    // Initialize weights and biases
    std::vector<float> W1(INPUT_DIM * HIDDEN_DIM), b1(HIDDEN_DIM, 0);
    std::vector<float> W2(HIDDEN_DIM * OUTPUT_DIM), b2(OUTPUT_DIM, 0);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& w : W1) w = dist(rng);
    for (auto& w : W2) w = dist(rng);

    // Allocate activation buffers
    std::vector<float> hidden(train_samples * HIDDEN_DIM);
    std::vector<float> logits(train_samples * OUTPUT_DIM);
    std::vector<float> probs(train_samples * OUTPUT_DIM);

    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Forward pass
        matmul(X_train.data(), W1.data(), hidden.data(), b1.data(), train_samples, INPUT_DIM, HIDDEN_DIM);
        relu(hidden.data(), hidden.size());
        matmul(hidden.data(), W2.data(), logits.data(), b2.data(), train_samples, HIDDEN_DIM, OUTPUT_DIM);
        std::copy(logits.begin(), logits.end(), probs.begin());
        softmax(probs.data(), train_samples, OUTPUT_DIM);

        // Compute dY (gradient of softmax + cross-entropy)
        std::vector<float> dY(train_samples * OUTPUT_DIM);
        for (int i = 0; i < train_samples; ++i)
            for (int j = 0; j < OUTPUT_DIM; ++j)
                dY[i * OUTPUT_DIM + j] = probs[i * OUTPUT_DIM + j] - (y_train[i] == j ? 1.0f : 0.0f);

        // Backpropagation for W2, b2
        std::vector<float> dW2(HIDDEN_DIM * OUTPUT_DIM, 0), db2(OUTPUT_DIM, 0);
        for (int i = 0; i < HIDDEN_DIM; ++i)
            for (int j = 0; j < OUTPUT_DIM; ++j)
                for (int n = 0; n < train_samples; ++n)
                    dW2[i * OUTPUT_DIM + j] += hidden[n * HIDDEN_DIM + i] * dY[n * OUTPUT_DIM + j] / train_samples;
        for (int j = 0; j < OUTPUT_DIM; ++j)
            for (int n = 0; n < train_samples; ++n)
                db2[j] += dY[n * OUTPUT_DIM + j] / train_samples;

        // Backpropagation for W1, b1
        std::vector<float> dW1(INPUT_DIM * HIDDEN_DIM, 0), db1(HIDDEN_DIM, 0);
        for (int i = 0; i < INPUT_DIM; ++i)
            for (int j = 0; j < HIDDEN_DIM; ++j)
                for (int n = 0; n < train_samples; ++n)
                    dW1[i * HIDDEN_DIM + j] += X_train[n * INPUT_DIM + i] * dY[n * OUTPUT_DIM + j] / train_samples;
        for (int j = 0; j < HIDDEN_DIM; ++j)
            for (int n = 0; n < train_samples; ++n)
                db1[j] += dY[n * OUTPUT_DIM + j] / train_samples;

        // Apply gradients with L2 regularization
        float lambda = 1e-4f;
        for (int i = 0; i < W1.size(); ++i) W1[i] -= LR * (dW1[i] + lambda * W1[i]);
        for (int i = 0; i < W2.size(); ++i) W2[i] -= LR * (dW2[i] + lambda * W2[i]);
        for (int i = 0; i < b1.size(); ++i) b1[i] -= LR * db1[i];
        for (int i = 0; i < b2.size(); ++i) b2[i] -= LR * db2[i];

        // Monitor training progress
        if (epoch % 100 == 0 || epoch == EPOCHS - 1) {
            float loss = cross_entropy_loss(probs.data(), y_train, train_samples, OUTPUT_DIM);
            float acc = compute_accuracy(probs.data(), y_train, train_samples);
            std::cout << "Epoch " << epoch << " - Loss: " << loss << ", Accuracy: " << acc << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\nCPU Training time: " << elapsed.count() << "s" << std::endl;

    // Evaluate on test set
    std::vector<float> hidden_test(test_samples * HIDDEN_DIM);
    std::vector<float> logits_test(test_samples * OUTPUT_DIM);
    std::vector<float> probs_test(test_samples * OUTPUT_DIM);
    matmul(X_test.data(), W1.data(), hidden_test.data(), b1.data(), test_samples, INPUT_DIM, HIDDEN_DIM);
    relu(hidden_test.data(), hidden_test.size());
    matmul(hidden_test.data(), W2.data(), logits_test.data(), b2.data(), test_samples, HIDDEN_DIM, OUTPUT_DIM);
    std::copy(logits_test.begin(), logits_test.end(), probs_test.begin());
    softmax(probs_test.data(), test_samples, OUTPUT_DIM);
    float test_acc = compute_accuracy(probs_test.data(), y_test, test_samples);
    std::cout << "Test Accuracy: " << test_acc << std::endl;

    return 0;
}
