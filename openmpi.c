/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un multicomputador usando OpenMPI

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Constantes
#define N 8
#define CLOCKS_PER_SEC 1000000

// Función para multiplicar matrices
void multiplicar_matrices(int **matriz_a, int **matriz_b, int **matriz_c) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			matriz_c[i][j] = 0;
			for (k = 0; k < N; k++) {
				matriz_c[i][j] += matriz_a[i][k] * matriz_b[k][j];
			}
		}
	}
}

// Función para imprimir matrices
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

// Función para rellenar matrices
void rellenar_matriz(int **matriz) {
	int i, j;
	for (i = 0; i < N; i++) {
		matriz[i] = (int *) malloc(N * sizeof(int));
		for (j = 0; j < N; j++) {
			matriz[i][j] = rand() % 10;
		}
	}
}

// Función principal
int main(int argc, char *argv[]) {
	// Declaración de variables
	int rank, size;

	// Inicializar MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Empezar contador de tiempo
	clock_t start = clock(); // CPU

	// Verificar que el número de procesos sea el correcto
	if (size != N) {
		if (rank == 0) {
			printf("Este programa está diseñado para funcionar %d procesos MPI.\n", N);
		}
		MPI_Finalize();
		return 1;
	}

	// Declaración de variables
	int **matrix_a, **matrix_b, **matrix_c;
	srand(time(NULL)); // semilla para generar números aleatorios

	if (rank == 0) {
		// Llenar las matrices A y B solo en el proceso 0
		rellenar_matriz(matrix_a);
		rellenar_matriz(matrix_b);

		printf("\nMatriz A:\n");
		imprimir_matriz(matrix_a);

		printf("\nMatriz B:\n");
		imprimir_matriz(matrix_b);
	}

	// Transmitir las matrices A y B a todos los procesos
	MPI_Bcast(&matrix_a[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&matrix_b[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);

	// Multiplicación de matrices
	multiplicar_matrices(matrix_a, matrix_b, matrix_c);

	// Recopilar resultados en el proceso 0
	MPI_Gather(&matrix_c[0][0], N * N, MPI_INT, &matrix_c[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("\nMatriz C (resultado):\n");
		imprimir_matriz(matrix_c);
	}

	// Imprimir tiempo de ejecución
	printf("\n-------------------\n");
	printf("Proceso %d de %d\n", rank, size);
	printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double) clock() - start) / CLOCKS_PER_SEC);

	// Finalizar MPI
	MPI_Finalize();

	// Fin del programa
	return 0;
}