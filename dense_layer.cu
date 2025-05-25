#include <cuda_runtime.h>
#include "dense_layer.h"

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

__global__ void relu_kernel(float* A, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}

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

__global__ void backward_kernel(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // input dim
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output dim

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
}

void dense_forward(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size) {
    dim3 block(16, 16);
    dim3 grid((out_size + 15) / 16, (batch + 15) / 16);
    forward_kernel<<<grid, block>>>(X, W, b, Y, batch, in_size, out_size);
    cudaDeviceSynchronize();
}

void dense_backward(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size) {
    dim3 block(16, 16);
    dim3 grid((out_size + 15) / 16, (in_size + 15) / 16);
    backward_kernel<<<grid, block>>>(dY, X, dW, db, batch, in_size, out_size);
    cudaDeviceSynchronize();
}

void relu_forward(float* A, int total) {
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(A, total);
    cudaDeviceSynchronize();
}

void softmax_forward(float* input, float* output, int batch, int dim) {
    int blockSize = 256;
    int gridSize = (batch + blockSize - 1) / blockSize;
    softmax_kernel<<<gridSize, blockSize>>>(input, output, batch, dim);
    cudaDeviceSynchronize();
}
