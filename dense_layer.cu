#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// leaky ReLU forward kernel
__global__ void leaky_relu_kernel(float* A, int total, float alpha) 
// float* A - pointer to a contiguous array of total floats in device memory
// int total - A element count
// float alpha - leak factor
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // compute global thread index
    if (idx < total) { // bound check
        A[idx] = (A[idx] > 0.0f) ? A[idx] : alpha * A[idx]; // keeps the positive value, multiplies the negative value with a small alpha
    }
}

// leaky ReLU backward kernel
__global__ void leaky_relu_backward_kernel(float* dY, float* hidden, int total, float alpha) 
// float* dY - pointer to the gradient of loss regarding layers outputs in device memory
// float* hidden - pointer to the pre-activation outputs (values before applying leaky ReLU) in device memory
// int total - element count
// float alpha - same leak factor
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total) {
            float derivative = (hidden[idx] > 0.0f ? 1.0f : alpha);
            dY[idx] *= derivative;
 }
}


// forward pass kernel for dense layer
__global__ void forward_kernel(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size)
// float* X       - pointer to input matrix [batch × in_size] in device memory
// float* W       - pointer to weight matrix [in_size × out_size] in device memory
// float* b       - pointer to bias vector [out_size] in device memory
// float* Y       - pointer to output matrix [batch × out_size] in device memory
// int   batch    - number of samples in the batch
// int   in_size  - number of input features per sample
// int   out_size - number of output features per sample
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // global batch index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // global output feature index

    if (row < batch && col < out_size) { // boundary check
        float sum = 0.0f; // accumulator for dot product
        for (int i = 0; i < in_size; ++i) { // iterate over input values and sum their multiplications with associated weights
            sum += X[row * in_size + i] // load one of the input values
                 * W[i * out_size + col]; // load the associated weight
        }
        Y[row * out_size + col] = sum + b[col]; // write output + add bias
    }
}


// backward_kernel - computes gradients for a dense (fully connected) layer during backpropagation
__global__ void backward_kernel(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size, bool use_leaky_relu = false, float alpha = 0.01f)
// float* dY - gradient of the loss w.r.t. the output of the dense layer [batch × out_size]
// float* X - input activations to the dense layer [batch × in_size]
// float* dW - output: gradient w.r.t. weights [in_size × out_size]
// float* db - output: gradient w.r.t. biases [out_size]
// int batch - number of samples in the batch
// int in_size - number of input features (neurons before the dense layer)
// int out_size - number of output features (neurons after the dense layer)
// bool use_leaky_relu - unused in this kernel (possibly for activation backward later)
// float alpha - leaky ReLU slope (unused here)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // input neuron index (row of weight matrix)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // output neuron index (column of weight matrix)

    // Compute dW[i][j]: gradient of the loss w.r.t. weight from input i to output j
    if (i < in_size && j < out_size) {
        float grad = 0.0f;
        for (int n = 0; n < batch; ++n) {
            // Accumulate gradient across batch:
            // dW[i][j] += X[n][i] * dY[n][j]
            grad += X[n * in_size + i] * dY[n * out_size + j];
        }
        dW[i * out_size + j] = grad / batch; // average over batch
    }

    // Compute db[j]: gradient of the loss w.r.t. bias at output neuron j
    if (i == 0 && j < out_size) { // only one thread per j handles db[j]
        float bias_grad = 0.0f;
        for (int n = 0; n < batch; ++n) {
            // db[j] += dY[n][j]
            bias_grad += dY[n * out_size + j];
        }
        db[j] = bias_grad / batch; // average over batch
    }
}


// softmax kernel (used after the dense layer)
__global__ void softmax_kernel(float* input, float* output, int batch, int dim)
// float* input - pointer to input logits matrix [batch × dim] in device memory
// float* output - pointer to output probabilities matrix [batch × dim] in device memory
// int batch - number of samples in the batch
// int dim - number of classes (features) per sample
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global sample index
    if (i >= batch) return; // boundary check
    // find the maximum logit in this sample to prevent overflow when calculating e^x
    float max_val = input[i * dim]; // start with first class
    for (int j = 1; j < dim; ++j) { // find the max logit
        max_val = fmaxf(max_val, input[i * dim + j]);
    }

    // compute exponentials shifted by max_val, and accumulate their sum
    float sum_exp = 0.0f;
    for (int j = 0; j < dim; ++j) {
        float e = expf(input[i * dim + j] - max_val);
        output[i * dim + j] = e; // store temporary 
        sum_exp += e; // accumulate 
    }

    // normalize
    for (int j = 0; j < dim; ++j) {
        output[i * dim + j] /= sum_exp;
    }
}

// loss_gradient_kernel - computes gradient of cross-entropy loss w.r.t. softmax output, and per-sample loss value
__global__ void loss_gradient_kernel(const float* probs, const int* labels, float* dY, float* loss, int batch, int classes)
// const float* probs - pointer to predicted class probabilities [batch × classes] from softmax (device memory)
// const int* labels - pointer to true class indices [batch] (device memory)
// float* dY - pointer to gradient output [batch × classes], i.e., dL/d(logits)
// float* loss - pointer to per-sample loss output [batch]
// int batch - number of samples
// int classes - number of classes per sample
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global sample index
    if (i >= batch) return; // boundary check
    int label = labels[i]; // get true class label for this sample
    // compute negative log likelihood loss for this sample:
    // loss = -log(p_label), where p_label is the predicted probability for the true class
    float sample_loss = -logf(probs[i * classes + label] + 1e-8f); // add small epsilon for numerical stability
    loss[i] = sample_loss; // store the loss

    // compute gradient of cross-entropy loss w.r.t. softmax input (probabilities):
    // dL/dy_j = y_j - 1 if j == label, else y_j
    for (int j = 0; j < classes; ++j) {
        float target = (j == label) ? 1.0f : 0.0f; // 1.0 only at true class index
        dY[i * classes + j] = probs[i * classes + j] - target; // gradient of loss w.r.t. probs
    }
}


// accuracy_kernel - computes per-sample accuracy (1 if prediction correct, 0 otherwise)
__global__ void accuracy_kernel(const float* probs, const int* labels, int* correct, int batch, int classes)
// const float* probs - pointer to predicted probabilities [batch × classes] from softmax
// const int* labels - pointer to ground truth class indices [batch]
// int* correct - output array [batch], where correct[i] = 1 if prediction matches label, else 0
// int batch - number of samples in the batch
// int classes - number of output classes
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // global sample index
    if (i >= batch) return; // boundary check

    // Find the class with the highest predicted probability (argmax)
    int max_idx = 0; // index of highest probability class
    float max_val = probs[i * classes]; // initial max value

    for (int j = 1; j < classes; ++j) {
        float val = probs[i * classes + j]; // probability for class j
        if (val > max_val) { // found a higher probability
            max_val = val;
            max_idx = j; // update max index
        }
    }

    // Compare predicted class (max_idx) with ground truth label
    correct[i] = (max_idx == labels[i]) ? 1 : 0; // store 1 if correct, else 0
}



// hidden_grad_kernel - computes gradient of the loss w.r.t. hidden layer output (before activation)
// dY_hidden[n, h] = sum over k of dY[n, k] * W2[h, k], i.e., backpropagation through dense layer
__global__ void hidden_grad_kernel(
    const float* __restrict__ dY,        // [batchSize × outputSize] - gradient from next layer
    const float* __restrict__ W2,        // [hiddenSize × outputSize] - weights of next layer
    float*             dY_hidden,        // [batchSize × hiddenSize] - output: gradient w.r.t. hidden layer output
    int batchSize,                       // number of samples
    int hiddenSize,                      // number of neurons in the hidden layer
    int outputSize                       // number of output neurons
)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x; // global flat index
    int total = batchSize * hiddenSize; // total number of (sample, hidden) pairs
    if (idx >= total) return; // boundary check

    int n = idx / hiddenSize; // sample index
    int h = idx % hiddenSize; // hidden neuron index

    // Row pointer to dY[n] (gradient vector from output layer)
    const float* dY_row = dY + n * outputSize;

    // Row pointer to W2[h] (weights connecting hidden neuron h to all outputs)
    const float* W2_row = W2 + h * outputSize;

    // Compute the dot product:
    // dY_hidden[n][h] = sum_k (dY[n][k] * W2[h][k])
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < outputSize; ++k) {
        sum += dY_row[k] * W2_row[k];
    }

    // Store the result
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