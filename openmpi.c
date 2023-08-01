/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un multicomputador usando OpenMPI

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Constantes
#define N 8
#define CLOCKS_PER_SEC 1000000

// Funciones
void rellenar_matriz(int matrix[N][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i][j] = rand() % 10;
		}
	}
}

void imprimir_matriz(int matrix[N][N]) {
	// Imprime la matriz
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void multiplicar_matrices(int a[N][N], int b[N][N], int c[N][N]) {
	// Multiplicación de dos matrices cuadradas
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			c[i][j] = 0;
			for (int k = 0; k < N; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

// Función principal
int main(int argc, char *argv[]) {
	// Declaración de variables
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Empezar contador de tiempo
	clock_t start = clock(); // CPU
	double start_mpi = MPI_Wtime(); // MPI

	if (size != N) {
		if (rank == 0) {
			printf("Este programa está diseñado para funcionar %d procesos MPI.\n", N);
		}
		MPI_Finalize();
		return 1;
	}

	int matrix_A[N][N], matrix_B[N][N], matrix_C[N][N];

	if (rank == 0) {
		// Llenar las matrices A y B solo en el proceso 0
		rellenar_matriz(matrix_A);
		rellenar_matriz(matrix_B);

		printf("\nMatriz A:\n");
		imprimir_matriz(matrix_A);

		printf("\nMatriz B:\n");
		imprimir_matriz(matrix_B);
	}

	// Transmitir las matrices A y B a todos los procesos
	MPI_Bcast(&matrix_A[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&matrix_B[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);

	// Multiplicación de matrices
	multiplicar_matrices(matrix_A, matrix_B, matrix_C);

	// Recopilar resultados en el proceso 0
	MPI_Gather(&matrix_C[0][0], N * N, MPI_INT, &matrix_C[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("\nMatriz C (resultado):\n");
		imprimir_matriz(matrix_C);
	}

	// Imprimir tiempo de ejecución
	printf("\nvvvvvvvvvvvv\n");
	printf("Proceso %d de %d\n", rank, size);
	printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double) clock() - start) / CLOCKS_PER_SEC);
	printf("Tiempo de ejecución del programa (MPI_Wtime): %f segundos\n", MPI_Wtime() - start_mpi);
	printf("^^^^^^^^^^^^\n");

	MPI_Finalize();

	return 0;
}

/// Instrucciones para compilar el programa:
/// mpicc openmpi.c -o openmpi
/// Instrucciones para ejecutar el programa:
/// mpirun -np X ./openmpi
/// X es el número de procesos a utilizar = N