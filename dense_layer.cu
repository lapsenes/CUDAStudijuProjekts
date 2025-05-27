#include <cuda_runtime.h>
#include "dense_layer.h"

// Matrix multiplication kernel with bias addition for a dense layer
__global__ void forward_kernel(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // sample index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // output neuron index

    if (row < batch && col < out_size) {
        float sum = 0.0f;
        for (int i = 0; i < in_size; ++i) {
            sum += X[row * in_size + i] * W[i * out_size + col];
        }
        Y[row * out_size + col] = sum + b[col];
    }
}

// ReLU activation kernel
__global__ void relu_kernel(float* A, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        A[idx] = fmaxf(0.0f, A[idx]);
    }
}

__global__ void relu_backward_kernel(
    float*       dY_hidden,  // [batch × HIDDEN_DIM]
    const float* hidden,     // pre‐ReLU activations [batch × HIDDEN_DIM]
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total && hidden[idx] <= 0.0f) {
        dY_hidden[idx] = 0.0f;
    }
}

// Softmax kernel, operates per row (sample)
__global__ void softmax_kernel(float* input, float* output, int batch, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row index
    if (i >= batch) return;

    // Find max for numerical stability
    float max_val = input[i * dim];
    for (int j = 1; j < dim; ++j) {
        max_val = fmaxf(max_val, input[i * dim + j]);
    }

    // Compute softmax
    float sum_exp = 0.0f;
    for (int j = 0; j < dim; ++j) {
        output[i * dim + j] = expf(input[i * dim + j] - max_val);
        sum_exp += output[i * dim + j];
    }
    for (int j = 0; j < dim; ++j) {
        output[i * dim + j] /= sum_exp;
    }
}

// Backpropagation for dense layer (computes dW and db)
__global__ void backward_kernel(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // input feature index
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // output neuron index

    if (i < in_size && j < out_size) {
        float grad = 0.0f;
        for (int n = 0; n < batch; ++n) {
            grad += X[n * in_size + i] * dY[n * out_size + j];
        }
        dW[i * out_size + j] = grad / batch;
    }

    // Compute db for each output neuron (only once per output neuron)
    if (i == 0 && j < out_size) {
        float bias_grad = 0.0f;
        for (int n = 0; n < batch; ++n) {
            bias_grad += dY[n * out_size + j];
        }
        db[j] = bias_grad / batch;
    }
}

__global__ void hidden_grad_kernel(
    const float* __restrict__ dY,  
    const float* __restrict__ W2,  
    float*             dY_hidden,   
    int batch,
    int hiddenSize,
    int outputSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hiddenSize;
    if (idx >= total) return;

    int n = idx / hiddenSize;   // sample index
    int h = idx % hiddenSize;   // hidden‐unit index

    const float* dY_row = dY       + n * outputSize;
    const float* W2_row = W2       + h * outputSize;

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < outputSize; ++k) {
        sum += dY_row[k] * W2_row[k];
    }
    dY_hidden[idx] = sum;
}



__global__ void loss_gradient_kernel(const float* probs, const int* labels, float* dY, float* loss, int batch, int classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;

    int label = labels[i];
    float sample_loss = -logf(probs[i * classes + label] + 1e-8f); // avoid log(0)
    loss[i] = sample_loss;

    for (int j = 0; j < classes; ++j) {
        dY[i * classes + j] = probs[i * classes + j] - (label == j ? 1.0f : 0.0f);
    }
}

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



// Launches forward kernel
void dense_forward(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size) {
    dim3 block(16, 16);
    dim3 grid((out_size + 15) / 16, (batch + 15) / 16);
    forward_kernel<<<grid, block>>>(X, W, b, Y, batch, in_size, out_size);
    cudaDeviceSynchronize();
}

// Launches backward kernel
void dense_backward(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size) {
    dim3 block(16, 16);
    dim3 grid((out_size + 15) / 16, (in_size + 15) / 16);
    backward_kernel<<<grid, block>>>(dY, X, dW, db, batch, in_size, out_size);
    cudaDeviceSynchronize();
}

// Launches ReLU activation kernel
void relu_forward(float* A, int total) {
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    relu_kernel<<<gridSize, blockSize>>>(A, total);
    cudaDeviceSynchronize();
}

// Launches softmax kernel
void softmax_forward(float* input, float* output, int batch, int dim) {
    int blockSize = 256;
    int gridSize = (batch + blockSize - 1) / blockSize;
    softmax_kernel<<<gridSize, blockSize>>>(input, output, batch, dim);
    cudaDeviceSynchronize();
}

void compute_loss_and_gradient_cuda(const float* probs, const int* labels, float* dY, float* loss_array, int batch, int classes) {
    int blockSize = 256;
    int gridSize = (batch + blockSize - 1) / blockSize;
    loss_gradient_kernel<<<gridSize, blockSize>>>(probs, labels, dY, loss_array, batch, classes);
    cudaDeviceSynchronize();
}

void compute_accuracy_cuda(const float* probs, const int* labels, int* correct_array, int batch, int classes) {
    int blockSize = 256;
    int gridSize = (batch + blockSize - 1) / blockSize;
    accuracy_kernel<<<gridSize, blockSize>>>(probs, labels, correct_array, batch, classes);
    cudaDeviceSynchronize();
}

void hidden_grad(
    const float* dY, const float* W2, float* dY_hidden,
    int batch, int hiddenSize, int outputSize
) {
    int total     = batch * hiddenSize;
    int blockSize = 256;
    int gridSize  = (total + blockSize - 1) / blockSize;
    hidden_grad_kernel<<<gridSize, blockSize>>>(
        dY, W2, dY_hidden, batch, hiddenSize, outputSize
    );
    cudaDeviceSynchronize();
}

void relu_backward(float* dY_hidden, const float* hidden, int total) {
    int blockSize = 256;
    int gridSize  = (total + blockSize - 1) / blockSize;
    relu_backward_kernel<<<gridSize, blockSize>>>(dY_hidden, hidden, total);
    cudaDeviceSynchronize();
}