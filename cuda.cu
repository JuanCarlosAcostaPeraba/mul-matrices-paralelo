/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Definiciones
#define N 2 // Tamaño de la matriz
#define BLOCK_SIZE 2 // Tamaño del bloque

// Función para multiplicar matrices en el host
void matrixMultCPU(int *a, int *b, int *c, int n) {
		int val = 0;
		for (int row = 0; row < n; row++) {
				for (int col = 0; col < n; col++){
						val = 0;
						for (int k = 0; k < n; k++) {
								val += a[row * n + k] * b[k * n + col];
						}
						c[row * n + col] = val;
				}
		}
}

// Función para multiplicar matrices en el device
__global__ void matrixMultGPU(int *a, int *b, int *c, int n) {
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int val = 0;
		if (row < n && col < n) {
				for (int k = 0; k < n; k++) {
						val += a[row * n + k] * b[k * n + col];
				}
				c[row * n + col] = val;
		}
}

// Función para imprimir matrices
void printMatrix(int *a, int n) {
		for (int row = 0; row < n; row++) {
				for (int col = 0; col < n; col++) {
						printf("%d ", a[row * n + col]);
				}
				printf("\n");
		}
}

// Función principal
int main() {
		// Declaración de variables
		int *a, *b, *c, *d, *e, *f; // Matrices en el host
		int *dev_a, *dev_b, *dev_c; // Matrices en el device
		int size = N * N * sizeof(int); // Tamaño de la matriz
		int i, j; // Variables auxiliares

		// Reserva de memoria en el host
		a = (int*)malloc(size);
		b = (int*)malloc(size);
		c = (int*)malloc(size);
		d = (int*)malloc(size);
		e = (int*)malloc(size);
		f = (int*)malloc(size);

		// Inicialización de matrices
		for (i = 0; i < N; i++) {
				for (j = 0; j < N; j++) {
						a[i * N + j] = rand() % 10;
						b[i * N + j] = rand() % 10;
						c[i * N + j] = 0;
						d[i * N + j] = rand() % 10;
						e[i * N + j] = rand() % 10;
						f[i * N + j] = 0;
				}
		}

		// Reserva de memoria en el device
		cudaMalloc((void**)&dev_a, size);
		cudaMalloc((void**)&dev_b, size);
		cudaMalloc((void**)&dev_c, size);

		// Copia de datos del host al device
		cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

		// Multiplicación de matrices en el host
		matrixMultCPU(a, b, c, N);

		// Multiplicación de matrices en el device
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		dim3 dimGrid((int)ceil(N / (float)dimBlock.x), (int)ceil(N / (float)dimBlock.y));
		matrixMultGPU<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);

		// Copia de datos del device al host
		cudaMemcpy(d, dev_c, size, cudaMemcpyDeviceToHost);

		// Impresión de matrices
		printf("Matriz A:\n");
		printMatrix(a, N);
		printf("Matriz B:\n");
		printMatrix(b, N);
		printf("Matriz C:\n");
		printMatrix(c, N);
		printf("Matriz D:\n");
		printMatrix(d, N);
		printf("Matriz E:\n");
		printMatrix(e, N);
		printf("Matriz F:\n");
		printMatrix(f, N);

		// Liberación de memoria en el host
		free(a);
		free(b);
		free(c);
		free(d);
		free(e);
		free(f);

		// Liberación de memoria en el device
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);

		return 0;
}
