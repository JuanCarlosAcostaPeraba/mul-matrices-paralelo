/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>
#include <stdlib.h>

// Constantes
#define N 2 // Tamaño de las matrices

// Funciones
__global__ void multiplicar_matrices(int **matriz_a, int **matriz_b, int **matriz_c) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N) {
		int sum = 0;
		for (int i = 0; i < N; i++) {
			sum += matriz_a[row][i] * matriz_b[i][col];
		}
		matriz_c[row][col] = sum;
	}
}

void imprimir_matriz(int **matriz) {
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%d ", matriz[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

// Función principal
int main(int argc, char *argv[]) {
	// Declaración de variables
	int i, j;
	int **matriz_a, **matriz_b, **matriz_c;

	struct timeval inicio, fin;

	// Empezar contador de tiempo
	clock_t start = clock(); // CPU
	gettimeofday(&inicio, NULL); // Hora del sistema

	// Reserva de memoria para las matrices
	matriz_a = (int **) malloc(N * sizeof(int *));
	matriz_b = (int **) malloc(N * sizeof(int *));
	matriz_c = (int **) malloc(N * sizeof(int *));
	for (i = 0; i < N; i++) {
		matriz_a[i] = (int *) malloc(N * sizeof(int));
		matriz_b[i] = (int *) malloc(N * sizeof(int));
		matriz_c[i] = (int *) malloc(N * sizeof(int));
	}

	// Inicialización de las matrices
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			matriz_a[i][j] = rand() % 10;
			matriz_b[i][j] = rand() % 10;
			matriz_c[i][j] = 0;
		}
	}

	// Imprimir matrices
	printf("Matriz A:\n");
	imprimir_matriz(matriz_a);
	printf("Matriz B:\n");
	imprimir_matriz(matriz_b);

	// Multiplicar matrices
	int **matriz_a_d, **matriz_b_d, **matriz_c_d;
	cudaMalloc((void **) &matriz_a_d, N * sizeof(int *));
	cudaMalloc((void **) &matriz_b_d, N * sizeof(int *));
	cudaMalloc((void **) &matriz_c_d, N * sizeof(int *));
	for (i = 0; i < N; i++) {
		cudaMalloc((void **) &matriz_a_d[i], N * sizeof(int));
		cudaMalloc((void **) &matriz_b_d[i], N * sizeof(int));
		cudaMalloc((void **) &matriz_c_d[i], N * sizeof(int));
	}
	cudaMemcpy(matriz_a_d, matriz_a, N * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(matriz_b_d, matriz_b, N * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(matriz_c_d, matriz_c, N * sizeof(int *), cudaMemcpyHostToDevice);

	dim3 dimGrid(1, 1);
	dim3 dimBlock(N, N);
	multiplicar_matrices<<<dimGrid, dimBlock>>>(matriz_a_d, matriz_b_d, matriz_c_d);

	cudaMemcpy(matriz_c, matriz_c_d, N * sizeof(int *), cudaMemcpyDeviceToHost);

	// Imprimir matriz resultante
	printf("Matriz C:\n");
	imprimir_matriz(matriz_c);

	// Liberar memoria
	for (i = 0; i < N; i++) {
		free(matriz_a[i]);
		free(matriz_b[i]);
		free(matriz_c[i]);
	}
	free(matriz_a);
	free(matriz_b);
	free(matriz_c);

	gettimeofday(&fin, NULL);

	// Imprimir tiempo de ejecución
	printf("\n-------------------\n");
	printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double) clock() - start) / CLOCKS_PER_SEC);
	printf("Tiempo de ejecución del programa (gettimeofday): %f segundos\n", (double) (fin.tv_sec - inicio.tv_sec) + (double) (fin.tv_usec - inicio.tv_usec) / 1000000);

	// Finalizar programa normalmente
	return 0;
}

/// Instrucciones para compilar el programa:
/// 
/// Instrucciones para ejecutar el programa:
/// 