/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>

// Constantes
#define N 2 // Tamaño de la matriz
#define BLOCK_SIZE 2 // Tamaño del bloque

// Funciones
// Kernel
__global__ void multiplicar_matrices(int *a, int *b, int *c) {
	// Calcula el índice de la matriz
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Calcula el elemento de la matriz
	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += a[row * N + i] * b[i * N + col];
	}

	// Guarda el resultado
	c[row * N + col] = sum;
}

// rellenar_matriz
void rellenar_matriz(int *matriz) {
	for (int i = 0; i < N * N; i++) {
		matriz[i] = rand() % 10;
	}
}

// imprimir_matriz
void imprimir_matriz(int *matriz) {
	for (int i = 0; i < N * N; i++) {
		printf("%d ", matriz[i]);
		if ((i + 1) % N == 0) {
			printf("\n");
		}
	}
}

// Función principal
int main() {
	// Matrices CPU
	int *a_cpu, *b_cpu, *c_cpu;
	// Matrices GPU
	int *a, *b, *c;
	size_t size = N * N * sizeof(int);

	// Reservar memoria en el CPU
	a_cpu = (int *)malloc(size);
	b_cpu = (int *)malloc(size);
	c_cpu = (int *)malloc(size);

	// Rellenar las matrices
	rellenar_matriz(a_cpu);
	rellenar_matriz(b_cpu);

	// Inicializar las matrices
	cudaMallocManaged(&a, size);
	cudaMallocManaged(&b, size);
	cudaMallocManaged(&c, size);

	// Copiar las matrices al GPU
	cudaMemcpy(a, a_cpu, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b, b_cpu, size, cudaMemcpyHostToDevice);

	// Definir las dimensiones del bloque y las hebras
	dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 num_blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

	// Ejecutar el kernel
	multiplicar_matrices<<<num_blocks, threads_per_block>>>(a, b, c);

	// Copiar el resultado al CPU
	cudaMemcpy(c_cpu, c, size, cudaMemcpyDeviceToHost);

	// Imprimir las matrices
	printf("Matriz A:\n");
	imprimir_matriz(a_cpu);
	printf("Matriz B:\n");
	imprimir_matriz(b_cpu);
	printf("Matriz C:\n");
	imprimir_matriz(c_cpu);

	// Liberar memoria
	free(a_cpu);
	free(b_cpu);
	free(c_cpu);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);


	return 0;
}