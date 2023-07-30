/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un multiprocesador usando OpenMPI

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Constantes
#define N 2
#define CLOCKS_PER_SEC 1000000

// Función para inicializar una matriz con valores aleatorios
void initializeMatrix(float** matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i][j] = (float)rand() / RAND_MAX; // Valores aleatorios entre 0 y 1
		}
	}
}


// Función para mostrar una matriz
void displayMatrix(float** matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%.2f\t", matrix[i][j]);
		}
		printf("\n");
	}
}

// Función para multiplicar las filas de una matriz por una matriz B
void multiplyRowsByMatrix(float** A, float** B, float** C, int rows_A, int cols_A, int cols_B) {
	for (int i = 0; i < rows_A; i++) {
		for (int j = 0; j < cols_B; j++) {
			C[i][j] = 0.0;
			for (int k = 0; k < cols_A; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

// Función principal
int main(int argc, char** argv) {
	int num_procs, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Tamaño de las matrices
	int rows_A = N; // Tamaño de filas de la matriz A
	int cols_A = N; // Tamaño de columnas de la matriz A
	int cols_B = N; // Tamaño de columnas de la matriz B

	// Verificar que el número de procesos sea mayor o igual al número de filas de la matriz A
	if (num_procs < rows_A) {
		if (rank == 0) {
			printf("Error: El número de procesos debe ser mayor o igual al número de filas de la matriz A.\n");
		}
		MPI_Finalize();
		return 1;
	}

	// Reservar memoria para las matrices A, B y C
	float** A = NULL;
	float** B = NULL;
	float** C = NULL;

	// El proceso 0 se encargará de la matriz A y la matriz resultante C
	if (rank == 0) {
			A = (float**)malloc(rows_A * sizeof(float*));
			C = (float**)malloc(rows_A * sizeof(float*));
			for (int i = 0; i < rows_A; i++) {
				A[i] = (float*)malloc(cols_A * sizeof(float));
				C[i] = (float*)malloc(cols_B * sizeof(float));
			}
			// Inicializar matriz A con valores aleatorios
			initializeMatrix(A, rows_A, cols_A);
			printf("Matriz A:\n");
			displayMatrix(A, rows_A, cols_A);
	}

	// Todos los procesos crean la matriz B
	B = (float**)malloc(cols_A * sizeof(float*));
	for (int i = 0; i < cols_A; i++) {
		B[i] = (float*)malloc(cols_B * sizeof(float));
	}

	// Inicializar matriz B con valores aleatorios en el proceso 0
	if (rank == 0) {
		initializeMatrix(B, cols_A, cols_B);
		printf("\nMatriz B:\n");
		displayMatrix(B, cols_A, cols_B);
	}

	// Todos los procesos calculan la multiplicación de sus filas de A por la matriz B
	float** local_C = (float**)malloc(rows_A * sizeof(float*));
	for (int i = 0; i < rows_A; i++) {
		local_C[i] = (float*)malloc(cols_B * sizeof(float));
	}

	// Comunicación de la matriz B a todos los procesos
	MPI_Bcast(&(B[0][0]), cols_A * cols_B, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Reparto de filas de A a los procesos
	int local_rows = rows_A / num_procs;
	int remaining_rows = rows_A % num_procs;
	int* send_counts = (int*)malloc(num_procs * sizeof(int));
	int* displs = (int*)malloc(num_procs * sizeof(int));

	for (int i = 0; i < num_procs; i++) {
		send_counts[i] = (i < remaining_rows) ? local_rows + 1 : local_rows;
		displs[i] = (i > 0) ? displs[i - 1] + send_counts[i - 1] : 0;
	}

	// Reparto de filas de A
	float* send_buffer = (float*)malloc(send_counts[rank] * cols_A * sizeof(float));
	MPI_Scatterv(&(A[0][0]), send_counts, displs, MPI_FLOAT, send_buffer, send_counts[rank] * cols_A, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// Cálculo de la multiplicación local
	float** local_A = (float**)malloc(send_counts[rank] * sizeof(float*));
	for (int i = 0; i < send_counts[rank]; i++) {
		local_A[i] = &(send_buffer[i * cols_A]);
	}

	multiplyRowsByMatrix(local_A, B, local_C, send_counts[rank], cols_A, cols_B);

	// Recopilación de los resultados parciales en el proceso 0
	MPI_Gatherv(&(local_C[0][0]), send_counts[rank] * cols_B, MPI_FLOAT, &(C[0][0]), send_counts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// El proceso 0 muestra el resultado final
	if (rank == 0) {
		printf("\nResultado de la multiplicación (Distribuido):\n");
		displayMatrix(C, rows_A, cols_B);
	}

	// Liberar memoria
	if (rank == 0) {
		for (int i = 0; i < rows_A; i++) {
			free(A[i]);
			free(C[i]);
		}
		free(A);
		free(C);
	}

	for (int i = 0; i < cols_A; i++) {
		free(B[i]);
	}
	free(B);

	for (int i = 0; i < send_counts[rank]; i++) {
		free(local_A[i]);
		free(local_C[i]);
	}
	free(local_A);
	free(local_C);
	free(send_buffer);
	free(send_counts);
	free(displs);

	MPI_Finalize();

	return 0;
}

/// Instrucciones para compilar el programa:
/// mpicc openmpi.c -o openmpi
/// Instrucciones para ejecutar el programa:
/// mpirun -np X ./openmpi
/// X es el número de procesos a utilizar