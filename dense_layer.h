#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

void dense_forward(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size);
void dense_backward(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size);
void relu_forward(float* A, int total);
void softmax_forward(float* input, float* output, int batch, int dim);

#endif
