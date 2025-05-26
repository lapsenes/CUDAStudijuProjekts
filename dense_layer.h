#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

void dense_forward(float* X, float* W, float* b, float* Y, int batch, int in_size, int out_size);
void dense_backward(float* dY, float* X, float* dW, float* db, int batch, int in_size, int out_size, bool use_leaky_relu = false, float alpha = 0.01f); 
void leaky_relu_forward(float* A, int total, float alpha = 0.01f);  
void softmax_forward(float* input, float* output, int batch, int dim);
void compute_loss_and_gradient_cuda(const float* probs, const int* labels, float* dY, float* loss_array, int batch, int classes);
void compute_accuracy_cuda(const float* probs, const int* labels, int* correct_array, int batch, int classes);
void leaky_relu_backward(float* dY, float* hidden, int total, float alpha = 0.01f);
void hidden_grad(const float* dY,const float* W2,float*       dY_hidden,int          batchSize,int          hiddenSize,int          outputSize);

#endif
