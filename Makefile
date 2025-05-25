all:
	nvcc -o main main.cu dense_layer.cu

clean:
	rm -f main
