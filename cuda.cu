/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>

// Constantes
#define N 1024
#define BLOCK_SIZE 32
#define CLOCKS_PER_SEC 1000000

// Función para multiplicar matrices
__global__ void multiplicar_matrices(int *matriz_a, int *matriz_b, int *matriz_c) {
	int fila = blockIdx.y * blockDim.y + threadIdx.y;
	int columna = blockIdx.x * blockDim.x + threadIdx.x;

	if (fila < N && columna < N) {
		int suma = 0;
		for (int i = 0; i < N; i++) {
			suma += matriz_a[fila * N + i] * matriz_b[i * N + columna];
		}
		matriz_c[fila * N + columna] = suma;
	}
}

// Función para imprimir matrices
void imprimir_matriz(int *matriz) {
	for (int i = 0; i < N * N; i++) {
		printf("%d ", matriz[i]);
		if ((i + 1) % N == 0) {
			printf("\n");
		}
	}
	printf("\n");
}

// Main
int main() {
	// Declaración de variables
	int *a_cpu, *b_cpu, *c_cpu; // Matrices CPU
	int *a_gpu, *b_gpu, *c_gpu; // Matrices GPU
	size_t size = N * N * sizeof(int); // Tamaño de las matrices
	srand(time(NULL)); // semilla para generar números aleatorios

	// Empezar contador de tiempo
	clock_t start = clock(); // CPU

	// Reserva de memoria en CPU
	a_cpu = (int *)malloc(size);
	b_cpu = (int *)malloc(size);
	c_cpu = (int *)malloc(size);

	// Inicializar matrices
	for (int i = 0; i < N * N; i++) {
		a_cpu[i] = rand() % 10;
		b_cpu[i] = rand() % 10;
	}

	// Imprimir matrices
	printf("Matriz A:\n");
	imprimir_matriz(a_cpu);
	printf("Matriz B:\n");
	imprimir_matriz(b_cpu);

	// Reserva de memoria en GPU
	cudaMalloc(&a_gpu, size);
	cudaMalloc(&b_gpu, size);
	cudaMalloc(&c_gpu, size);

	// Copiar datos de CPU a GPU
	cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);

	// Inicializar matriz C en GPU
	memset(c_cpu, 0, size);

	// Definir bloques e hilos para el kernel
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	// Lanzar kernel
	multiplicar_matrices<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, N);

	// Sincronizar kernel
	cudaDeviceSynchronize();

	// Verificar errores
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error de CUDA: %s\n", cudaGetErrorString(error));
	}

	// Copiar datos de GPU a CPU
	cudaMemcpy(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost);

	// Mostrar resultado
	printf("Matriz C:\n");
	imprimir_matriz(c_cpu);

	// Liberar memoria
	free(a_cpu);
	free(b_cpu);
	free(c_cpu);
	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(c_gpu);

	// Imprimir tiempo de ejecución
	printf("\n-------------------\n");
	printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double) clock() - start) / CLOCKS_PER_SEC);

	// Fin del programa
	return 0;
}