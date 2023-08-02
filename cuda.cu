/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Definiciones
#define N 2 // Tamaño de las matrices

// Funciones
__global__ void multiplicar_matrices(int *matriz_a, int *matriz_b, int *matriz_c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += matriz_a[row * N + i] * matriz_b[i * N + col];
        }
        matriz_c[row * N + col] = sum;
    }
}

void imprimir_matriz(int *matriz) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matriz[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Main
int main() {
    int i, j;
    int matriz_a[N * N], matriz_b[N * N], matriz_c[N * N];

    // Inicialización de las matrices
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matriz_a[i * N + j] = rand() % 10;
            matriz_b[i * N + j] = rand() % 10;
            matriz_c[i * N + j] = 0;
        }
    }

    // Imprimir matrices
    printf("Matriz A:\n");
    imprimir_matriz(matriz_a);
    printf("Matriz B:\n");
    imprimir_matriz(matriz_b);

    // Multiplicar matrices en la GPU
    int *matriz_a_d, *matriz_b_d, *matriz_c_d;
    cudaMalloc((void**)&matriz_a_d, N * N * sizeof(int));
    cudaMalloc((void**)&matriz_b_d, N * N * sizeof(int));
    cudaMalloc((void**)&matriz_c_d, N * N * sizeof(int));

    cudaMemcpy(matriz_a_d, matriz_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matriz_b_d, matriz_b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1);
    dim3 dimBlock(N, N);
    multiplicar_matrices<<<dimGrid, dimBlock>>>(matriz_a_d, matriz_b_d, matriz_c_d);

    cudaMemcpy(matriz_c, matriz_c_d, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir matriz resultante
    printf("Matriz C:\n");
    imprimir_matriz(matriz_c);

    // Liberar memoria en la GPU
    cudaFree(matriz_a_d);
    cudaFree(matriz_b_d);
    cudaFree(matriz_c_d);

    return 0;
}