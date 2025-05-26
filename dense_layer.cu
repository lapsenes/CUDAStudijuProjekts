#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// leaky ReLU layer
__global__ void leaky_relu_kernel(float* A, int total, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // compute global thread index
    if (idx < total) { // bound check
        A[idx] = (A[idx] > 0.0f) ? A[idx] : alpha * A[idx]; // keeps the positive value, multiplies the negative value with a small alpha
    }
}

// Forward pass kernel for dense layer
__global__ void forward_kernel(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < batch && col < out_size) {
        float sum = 0.0f;
        for (int i = 0; i < in_size; ++i) {
            sum += X[row * in_size + i] * W[i * out_size + col];
        }
        Y[row * out_size + col] = sum + b[col];
    }
}

// Leaky ReLU backward kernel
__global__ void leaky_relu_backward_kernel(float* dY, float* hidden, int total, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        if (hidden[idx] <= 0.0f) {
            dY[idx] *= alpha;  // If input <= 0, scale the gradient by alpha
        }
    }
}

// Backward pass kernel for dense layer (computing dW and db)
__global__ void backward_kernel(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size, bool use_leaky_relu = false, float alpha = 0.01f) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // input feature index
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron index

    if (i < in_size && j < out_size) {
        float grad = 0.0f;
        for (int n = 0; n < batch; ++n) {
            grad += X[n * in_size + i] * dY[n * out_size + j];
        }
        dW[i * out_size + j] = grad / batch;
    }

    if (i == 0 && j < out_size) {
        float bias_grad = 0.0f;
        for (int n = 0; n < batch; ++n) {
            bias_grad += dY[n * out_size + j];
        }
        db[j] = bias_grad / batch;
    }

    // If using Leaky ReLU, adjust the gradient calculation
    if (use_leaky_relu && i < in_size && j < out_size) {
        // Leaky ReLU gradient adjustment: if the input was <= 0, multiply the gradient by alpha
        if (X[i] <= 0.0f) {
            dY[i] *= alpha;  // Adjust the gradient for the dead neurons (Leaky ReLU)
        }
    }
}

// Softmax kernel (used after the output layer)
__global__ void softmax_kernel(float* input, float* output, int batch, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= batch) return;

    float max_val = input[i * dim];
    for (int j = 1; j < dim; ++j) {
        max_val = fmaxf(max_val, input[i * dim + j]);
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < dim; ++j) {
        output[i * dim + j] = expf(input[i * dim + j] - max_val);
        sum_exp += output[i * dim + j];
    }
    for (int j = 0; j < dim; ++j) {
        output[i * dim + j] /= sum_exp;
    }
}

// Loss gradient kernel for cross-entropy loss and backpropagation
__global__ void loss_gradient_kernel(const float* probs, const int* labels, float* dY, float* loss, int batch, int classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;

    int label = labels[i];
    float sample_loss = -logf(probs[i * classes + label] + 1e-8f); // Log with small epsilon for numerical stability
    loss[i] = sample_loss;

    for (int j = 0; j < classes; ++j) {
        dY[i * classes + j] = probs[i * classes + j] - (label == j ? 1.0f : 0.0f);
    }
}

// Accuracy kernel
__global__ void accuracy_kernel(const float* probs, const int* labels, int* correct, int batch, int classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;

    int max_idx = 0;
    float max_val = probs[i * classes];
    for (int j = 1; j < classes; ++j) {
        float val = probs[i * classes + j];
        if (val > max_val) {
            max_val = val;
            max_idx = j;
        }
    }
    correct[i] = (max_idx == labels[i]) ? 1 : 0;
}


// Kernel: dY_hidden[n, h] = sum_k dY[n, k] * W2[h, k]
__global__ void hidden_grad_kernel(
    const float* __restrict__ dY,        // [batchSize, outputSize]
    const float* __restrict__ W2,        // [hiddenSize, outputSize]
    float*             dY_hidden,        // [batchSize, hiddenSize]
    int batchSize,
    int hiddenSize,
    int outputSize
) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * hiddenSize;
    if (idx >= total) return;

    int n = idx / hiddenSize;  // sample index
    int h = idx % hiddenSize;  // hidden feature index

    // pointers to row n of dY, row h of W2
    const float* dY_row = dY       + n * outputSize;
    const float* W2_row = W2       + h * outputSize;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < outputSize; ++k) {
        sum += dY_row[k] * W2_row[k];
    }

    dY_hidden[idx] = sum;
}







// Forward pass function for dense layer
void dense_forward(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size) {
    dim3 block(16, 16);
    dim3 grid((out_size + 15) / 16, (batch + 15) / 16);
    forward_kernel<<<grid, block>>>(X, W, b, Y, batch, in_size, out_size);
    cudaDeviceSynchronize();
}

// Backward pass function for dense layer
void dense_backward(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size, bool use_leaky_relu = false, float alpha = 0.01f) {
    dim3 block(16, 16);
    dim3 grid((out_size + 15) / 16, (in_size + 15) / 16);
    backward_kernel<<<grid, block>>>(dY, X, dW, db, batch, in_size, out_size, use_leaky_relu, alpha);
    cudaDeviceSynchronize();
}

// Leaky ReLU forward function
void leaky_relu_forward(float* A, int total, float alpha = 0.01f) {
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    leaky_relu_kernel<<<gridSize, blockSize>>>(A, total, alpha);
    cudaDeviceSynchronize();
}

// Softmax forward function
void softmax_forward(float* input, float* output, int batch, int dim) {
    int blockSize = 256;
    int gridSize = (batch + blockSize - 1) / blockSize;
    softmax_kernel<<<gridSize, blockSize>>>(input, output, batch, dim);
    cudaDeviceSynchronize();
}

// Loss and gradient computation function
void compute_loss_and_gradient_cuda(const float* probs, const int* labels, float* dY, float* loss_array, int batch, int classes) {
    int blockSize = 256;
    int gridSize = (batch + blockSize - 1) / blockSize;
    loss_gradient_kernel<<<gridSize, blockSize>>>(probs, labels, dY, loss_array, batch, classes);
    cudaDeviceSynchronize();
}

// Accuracy computation function
void compute_accuracy_cuda(const float* probs, const int* labels, int* correct_array, int batch, int classes) {
    int blockSize = 256;
    int gridSize = (batch + blockSize - 1) / blockSize;
    accuracy_kernel<<<gridSize, blockSize>>>(probs, labels, correct_array, batch, classes);
    cudaDeviceSynchronize();
}

// Leaky ReLU backward function
void leaky_relu_backward(float* dY, float* hidden, int total, float alpha = 0.01f) {
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    leaky_relu_backward_kernel<<<gridSize, blockSize>>>(dY, hidden, total, alpha);
    cudaDeviceSynchronize();
}


void hidden_grad(const float* dY, const float* W2, float* dY_hidden, int batchSize, int hiddenSize, int outputSize) {
    int total     = batchSize * hiddenSize;
    int blockSize = 256;
    int gridSize  = (total + blockSize - 1) / blockSize;
    hidden_grad_kernel<<<gridSize, blockSize>>>(dY, W2, dY_hidden, batchSize, hiddenSize, outputSize);
    cudaDeviceSynchronize();
}