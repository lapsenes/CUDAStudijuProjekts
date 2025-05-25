#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include "dense_layer.h"

#define INPUT_DIM   784    // 28x28 images for MNIST
#define HIDDEN_DIM  128
#define OUTPUT_DIM  10
#define EPOCHS      100
#define LR          0.0001f
#define TRAIN_RATIO 0.8f
#define CLIP_VALUE  5.0f
#define LAMBDA      1e-4f

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " '" #call "' failed: " << cudaGetErrorString(err) << "\n"; \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


  bool read_mnist_csv(const std::string& filename, std::vector<float>& X, std::vector<int>& y) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;

    // Skip the header line once
    if (!std::getline(file, line)) 
        return false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;  // Skip blank lines
        std::stringstream ss(line);
        std::string val;
        int col = 0;
        std::vector<float> sample(INPUT_DIM);  // Assuming INPUT_DIM is the number of pixel columns
        int label = 0;

        while (std::getline(ss, val, ',')) {
            if (col == 0) {  // Label is the first column
                try {
                    label = std::stoi(val);
                } catch (...) {
                    label = 0;  // Handle invalid label values
                }
            } else {  // Features are the remaining columns
                try {
                    sample[col - 1] = std::stof(val) / 255.0f;  // Normalize pixel values between 0 and 1
                } catch (...) {
                    sample[col - 1] = 0.0f;  // Handle invalid pixel values
                }
            }
            ++col;
        }

        // Add the sample to the data vector X and the label to y
        X.insert(X.end(), sample.begin(), sample.end());
        y.push_back(label);
    }

    return true;
}



void test_model(float* X_test, float* W1, float* b1, float* W2, float* b2,
                int* y_test, int test_samples,
                int input_dim, int hidden_dim, int output_dim) {
    float *hidden, *logits, *probs;
    CUDA_CHECK(cudaMallocManaged(&hidden, test_samples * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&logits, test_samples * output_dim * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&probs, test_samples * output_dim * sizeof(float)));

    // Forward pass
    dense_forward(X_test, W1, b1, hidden, test_samples, input_dim, hidden_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    leaky_relu_forward(hidden, test_samples * hidden_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    dense_forward(hidden, W2, b2, logits, test_samples, hidden_dim, output_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    softmax_forward(logits, probs, test_samples, output_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Accuracy
    int* correct_array;
    CUDA_CHECK(cudaMallocManaged(&correct_array, test_samples * sizeof(int)));
    compute_accuracy_cuda(probs, y_test, correct_array, test_samples, output_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    int correct = 0;
    for (int i = 0; i < test_samples; ++i) correct += correct_array[i];
    std::cout << "Test Accuracy: " << (float)correct / test_samples << std::endl;

    // Clean up
    cudaFree(hidden);
    cudaFree(logits);
    cudaFree(probs);
    cudaFree(correct_array);
}

int main() {
    std::vector<float> full_X;
    std::vector<int>   full_y;
    if (!read_mnist_csv("data/mnist.csv", full_X, full_y)) {
        std::cerr << "Failed to load mnist.csv\n";
        return 1;
    }

    

    int total_samples = full_y.size();
    int train_samples = int(total_samples * TRAIN_RATIO);
    int test_samples  = total_samples - train_samples;

    // Build dataset
    std::vector<std::pair<std::vector<float>,int>> dataset(total_samples);
    for (int i = 0; i < total_samples; ++i) {
        dataset[i].first.assign(
            full_X.begin() + i*INPUT_DIM,
            full_X.begin() + (i+1)*INPUT_DIM
        );
        dataset[i].second = full_y[i];
    }

    // Manual Fisher–Yates shuffle
    std::srand(static_cast<unsigned>(time(nullptr)));
    for (int i = total_samples - 1; i > 0; --i) {
        int j = std::rand() % (i + 1);
        std::swap(dataset[i], dataset[j]);
    }

    // Split X/y
    std::vector<float> X_train(train_samples*INPUT_DIM), X_test_vec(test_samples*INPUT_DIM);
    std::vector<int>   y_train(train_samples),       y_test(test_samples);
    for (int i = 0; i < train_samples; ++i) {
        std::copy(dataset[i].first.begin(), dataset[i].first.end(),
                  X_train.begin() + i*INPUT_DIM);
        y_train[i] = dataset[i].second;
    }
    for (int i = 0; i < test_samples; ++i) {
        std::copy(dataset[train_samples+i].first.begin(),
                  dataset[train_samples+i].first.end(),
                  X_test_vec.begin() + i*INPUT_DIM);
        y_test[i] = dataset[train_samples+i].second;
    }

    // Normalize
    for (int j = 0; j < INPUT_DIM; ++j) {
        double sum = 0, sum_sq = 0;
        for (int i = 0; i < train_samples; ++i) {
            float v = X_train[i*INPUT_DIM + j];
            sum += v; sum_sq += v*v;
        }
        float mean = sum / train_samples;
        float stdv = std::sqrt(sum_sq / train_samples - mean*mean + 1e-8f);
        for (int i = 0; i < train_samples; ++i)
            X_train[i*INPUT_DIM + j] = (X_train[i*INPUT_DIM + j] - mean) / stdv;
        for (int i = 0; i < test_samples; ++i)
            X_test_vec[i*INPUT_DIM + j] = (X_test_vec[i*INPUT_DIM + j] - mean) / stdv;
    }

    // Allocate model parameters
    float *W1, *b1, *W2, *b2;
    CUDA_CHECK(cudaMallocManaged(&W1, INPUT_DIM*HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b1, HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&W2, HIDDEN_DIM*OUTPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b2, OUTPUT_DIM*sizeof(float)));

    // He initialization
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-0.1f,0.1f);
    float scale = std::sqrt(2.0f/INPUT_DIM);
    for (int i = 0; i < INPUT_DIM*HIDDEN_DIM; ++i) W1[i] = dist(rng)*scale;
    for (int i = 0; i < HIDDEN_DIM; ++i) b1[i] = 0;
    for (int i = 0; i < HIDDEN_DIM*OUTPUT_DIM; ++i) W2[i] = dist(rng)*scale;
    for (int i = 0; i < OUTPUT_DIM; ++i) b2[i] = 0;

    // Allocate training buffers
    float *X, *hidden, *logits, *probs, *dY;
    int   *d_y_train;
    CUDA_CHECK(cudaMallocManaged(&X,      train_samples*INPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&hidden, train_samples*HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&logits, train_samples*OUTPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&probs,  train_samples*OUTPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&dY,     train_samples*OUTPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_y_train, train_samples*sizeof(int)));

    std::copy(X_train.begin(), X_train.end(), X);
    std::copy(y_train.begin(), y_train.end(), d_y_train);

    // Gradients & loss arrays
    float *dW1, *db1, *dW2, *db2, *loss_array;
    int   *correct_array;
    CUDA_CHECK(cudaMallocManaged(&dW1,       INPUT_DIM*HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&db1,       HIDDEN_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&dW2,       HIDDEN_DIM*OUTPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&db2,       OUTPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&loss_array,train_samples*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&correct_array, train_samples*sizeof(int)));

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Forward
        dense_forward(X,  W1, b1, hidden, train_samples, INPUT_DIM, HIDDEN_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        leaky_relu_forward(hidden, train_samples*HIDDEN_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        dense_forward(hidden, W2, b2, logits, train_samples, HIDDEN_DIM, OUTPUT_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());
        softmax_forward(logits, probs, train_samples, OUTPUT_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Loss & gradient
        compute_loss_and_gradient_cuda(probs, d_y_train, dY, loss_array, train_samples, OUTPUT_DIM);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute epoch loss
        double loss = 0;
        for (int i = 0; i < train_samples; ++i) loss += loss_array[i];
        loss /= train_samples;

        // Backprop second layer
        dense_backward(dY, hidden, dW2, db2, train_samples, HIDDEN_DIM, OUTPUT_DIM, true);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Hidden grad on CPU + Leaky back
        float* dY_hidden;
        CUDA_CHECK(cudaMallocManaged(&dY_hidden, train_samples*HIDDEN_DIM*sizeof(float)));
        for (int i = 0; i < train_samples; ++i)
          for (int j = 0; j < HIDDEN_DIM; ++j) {
            float sum = 0;
            for (int k = 0; k < OUTPUT_DIM; ++k)
              sum += dY[i*OUTPUT_DIM + k]*W2[j*OUTPUT_DIM + k];
            dY_hidden[i*HIDDEN_DIM + j] = sum;
          }
        leaky_relu_backward(dY_hidden, hidden, train_samples*HIDDEN_DIM, 0.01f);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Backprop first layer
        dense_backward(dY_hidden, X, dW1, db1, train_samples, INPUT_DIM, HIDDEN_DIM, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(dY_hidden);

        // Update weights
        for (int i = 0; i < INPUT_DIM*HIDDEN_DIM; ++i)
          W1[i] -= LR*(fminf(fmaxf(dW1[i], -CLIP_VALUE), CLIP_VALUE) + LAMBDA*W1[i]);
        for (int i = 0; i < HIDDEN_DIM; ++i)
          b1[i] -= LR*fminf(fmaxf(db1[i], -CLIP_VALUE), CLIP_VALUE);
        for (int i = 0; i < HIDDEN_DIM*OUTPUT_DIM; ++i)
          W2[i] -= LR*(fminf(fmaxf(dW2[i], -CLIP_VALUE), CLIP_VALUE) + LAMBDA*W2[i]);
        for (int i = 0; i < OUTPUT_DIM; ++i)
          b2[i] -= LR*fminf(fmaxf(db2[i], -CLIP_VALUE), CLIP_VALUE);

        if (epoch % 10 == 0 || epoch == EPOCHS-1) {
          compute_accuracy_cuda(probs, d_y_train, correct_array, train_samples, OUTPUT_DIM);
          CUDA_CHECK(cudaDeviceSynchronize());
          int correct = 0;
          for (int i = 0; i < train_samples; ++i) correct += correct_array[i];
          std::cout << "Epoch " << epoch
                    << " - Loss: " << loss
                    << ", Accuracy: " << double(correct)/train_samples
                    << "\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Training Time: "
              << std::chrono::duration<double>(t1-t0).count() << "s\n";

    // Test
    float* X_test_dev;  int* y_test_dev;
    CUDA_CHECK(cudaMallocManaged(&X_test_dev, test_samples*INPUT_DIM*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&y_test_dev, test_samples*sizeof(int)));
    std::copy(X_test_vec.begin(), X_test_vec.end(), X_test_dev);
    std::copy(y_test.begin(),    y_test.end(),    y_test_dev);

    test_model(X_test_dev, W1, b1, W2, b2, y_test_dev,
               test_samples, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);

    // Cleanup
    cudaFree(X_test_dev); cudaFree(y_test_dev);
    cudaFree(X); cudaFree(hidden); cudaFree(logits); cudaFree(probs);
    cudaFree(dY); cudaFree(d_y_train);
    cudaFree(dW1); cudaFree(db1); cudaFree(dW2); cudaFree(db2);
    cudaFree(loss_array); cudaFree(correct_array);
    cudaFree(W1); cudaFree(b1); cudaFree(W2); cudaFree(b2);

    return 0;
}
