/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Definiciones
#define N 2
#define BLOCK_SIZE 2

// Funciones
__global__ void multiplicar_matrices(int *a, int *b, int *c, int n) {
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	if (fila < n && columna < n) {
		int suma = 0;
		for (int i = 0; i < n; i++) {
			suma += a[fila * n + i] * b[i * n + columna];
		}
		c[fila * n + columna] = suma;
	}
}


// Main
int main() {
	int *a_cpu, *b_cpu, *c_cpu;
	int *a_gpu, *b_gpu, *c_gpu;
	size_t size = N * N * sizeof(int);

	// Reserva de memoria en CPU
	a_cpu = (int *)malloc(size);
	b_cpu = (int *)malloc(size);
	c_cpu = (int *)malloc(size);

	// Inicializar matrices
	for (int i = 0; i < N * N; i++) {
		a_cpu[i] = rand() % 10;
		b_cpu[i] = rand() % 10;
	}

	// Mostrar matrices
	printf("Matriz A:\n");
	for (int i = 0; i < N * N; i++) {
		printf("%d ", a_cpu[i]);
		if ((i + 1) % N == 0) {
			printf("\n");
		}
	}
	printf("\n");

	printf("Matriz B:\n");
	for (int i = 0; i < N * N; i++) {
		printf("%d ", b_cpu[i]);
		if ((i + 1) % N == 0) {
			printf("\n");
		}
	}
	printf("\n");

	// Reserva de memoria en GPU
	cudaMalloc(&a_gpu, size);
	cudaMalloc(&b_gpu, size);
	cudaMalloc(&c_gpu, size);

	// Copiar datos de CPU a GPU
	cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);

	// Definir bloques e hilos
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

	// Lanzar kernel
	multiplicar_matrices<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, N);

	// Copiar datos de GPU a CPU
	cudaMemcpy(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost);

	// Mostrar resultado
	printf("Matriz C:\n");
	for (int i = 0; i < N * N; i++) {
		printf("%d ", c_cpu[i]);
		if ((i + 1) % N == 0) {
			printf("\n");
		}
	}
	printf("\n");

	// Liberar memoria
	free(a_cpu);
	free(b_cpu);
	free(c_cpu);
	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(c_gpu);

	return 0;
}