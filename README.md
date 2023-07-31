## Objetivo

Implementar el algoritmo de multiplicación de matrices con números en coma flotante en las librerías paralelas OpenMP, OpenMPI y CUDA utilizando un ordenador sobremesa.

## Tareas a realizar

- [x] Implementación en un multiprocesador usando OpenMP
- [x] Implementación en un multicomputador usando OpenMPI
- [ ] Implementación en un coprocesador de tipo GPU usando CUDA
- [x] Evaluación de prestaciones usando contadores hardware
- [ ] Comparación de prestaciones entre multiprocesadores, multicomputadores y GPUs

## Herramientas informáticas

- Librería OpenMP
- Librería OpenMPI
- Compilador y librería CUDA
- Ordenador sobremesa

---

# Explicación

## Implementación secuencial

### Código

```c
// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constantes
#define N 1000
#define CLOCKS_PER_SEC 1000000

// Funciones
void multiplicar_matrices(int **matriz_a, int **matriz_b, int **matriz_c) {
	int i, j, k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				matriz_c[i][j] += matriz_a[i][k] * matriz_b[k][j];
			}
		}
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
	int **matriz_a, **matriz_b, **matriz_c;
	int i, j;

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

	// Inicializar matrices
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			// rellena la matriz a y b con valores aleatorios entre 0 y 9
			matriz_a[i][j] = rand() % 10;
			matriz_b[i][j] = rand() % 10;
			// inicializa la matriz c con 0
			matriz_c[i][j] = 0;
		}
	}

	// Multiplicar matrices
	multiplicar_matrices(matriz_a, matriz_b, matriz_c);

	// Imprimir matrices
	imprimir_matriz(matriz_a);
	imprimir_matriz(matriz_b);
	imprimir_matriz(matriz_c);

	// Liberar memoria de las matrices
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

	// Fin del programa
	return 0;
}
```

### Explicación

En este programa se realiza la multiplicación de dos matrices cuadradas de tamaño `N = 1000` de manera secuencial con un bucle `for`.

El tiempo que ha tardado este programa en ejecutarse en mi ordenador ha sido de:

```
-------------------
Tiempo de ejecución del programa (CPU): 3.158866 segundos
Tiempo de ejecución del programa (gettimeofday): 3.390733 segundos
```

### Como ejecutar

Para ejecutar el programa se necesita un compilador del lenguaje C instalado en la máquina y seguir los siguientes pasos:

1. Compilar el código

   ```bash
   gcc secuencial.c -o secuencial
   ```

2. Ejecutar el código

   ```bash
   ./secuencial
   ```

## Implementación en un multiprocesador usando OpenMP

### Código

```c
// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Constantes
#define N 1000
#define THREADS 6
#define CLOCKS_PER_SEC 1000000

// Funciones
void multiplicar_matrices(int **matriz_a, int **matriz_b, int **matriz_c) {
	int i, j, k;
	omp_set_num_threads(THREADS);
	#pragma omp parallel for private(i, j, k)
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				matriz_c[i][j] += matriz_a[i][k] * matriz_b[k][j];
			}
		}
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
	int **matriz_a, **matriz_b, **matriz_c;
	int i, j;

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

	// Inicializar matrices
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			// rellena la matriz a y b con valores aleatorios entre 0 y 9
			matriz_a[i][j] = rand() % 10;
			matriz_b[i][j] = rand() % 10;
			// inicializa la matriz c con ceros
			matriz_c[i][j] = 0;
		}
	}

	// Multiplicar matrices
	multiplicar_matrices(matriz_a, matriz_b, matriz_c);

	// Imprimir matrices
	imprimir_matriz(matriz_a);
	imprimir_matriz(matriz_b);
	imprimir_matriz(matriz_c);

	// Liberar memoria de las matrices
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
```

### Explicación

En este programa se realiza la multiplicación de dos matrices cuadradas de tamaño `N = 1000` de forma paralela en `hilos = 6` con la librería `OpenMP`.

El tiempo que ha tardado este programa en ejecutarse en mi ordenador ha sido de:

```
-------------------
Tiempo de ejecución del programa (CPU): 4.303740 segundos
Tiempo de ejecución del programa (gettimeofday): 1.187150 segundos
```

### Como ejecutar

Para ejecutar el programa se necesita un compilador del lenguaje C instalado en la máquina y la librería de `OpenMP`, y seguir los siguientes pasos:

1. Compilar el código

   ```bash
   gcc openmp.c -o openmp -fopenmp
   ```

2. Ejecutar el código

   ```bash
   ./openmp
   ```

## Implementación en un multicomputador usando OpenMPI

### Código

```c
// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Constantes
#define N 8
#define CLOCKS_PER_SEC 1000000

// Funciones
void fill_matrix(int matrix[N][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrix[i][j] = rand() % 10;
		}
	}
}

void print_matrix(int matrix[N][N]) {
	// Imprime la matriz
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void matrix_mult(int a[N][N], int b[N][N], int c[N][N]) {
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
		fill_matrix(matrix_A);
		fill_matrix(matrix_B);

		printf("\nMatriz A:\n");
		print_matrix(matrix_A);

		printf("\nMatriz B:\n");
		print_matrix(matrix_B);
	}

	// Transmitir las matrices A y B a todos los procesos
	MPI_Bcast(&matrix_A[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&matrix_B[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);

	// Multiplicación de matrices
	matrix_mult(matrix_A, matrix_B, matrix_C);

	// Recopilar resultados en el proceso 0
	MPI_Gather(&matrix_C[0][0], N * N, MPI_INT, &matrix_C[0][0], N * N, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("\nMatriz C (resultado):\n");
		print_matrix(matrix_C);
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
```

### Explicación

En este programa se realiza la multiplicación de dos matrices cuadradas de tamaño `N = 8` de forma paralela tratando cada `core` del procesador como un ordenador independiente. Por este motivo, el tamaño de las matrices no pueden ser mayores, ya que no dispongo de más núcleos.

El tiempo individual que ha tardado este programa en ejecutarse en mi ordenador ha sido de:

```
-------------------
Proceso 3 de 8
Tiempo de ejecución del programa (CPU): 0.000173 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000249 segundos
-------------------
Proceso 5 de 8
Tiempo de ejecución del programa (CPU): 0.000182 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000280 segundos
-------------------
Proceso 7 de 8
Tiempo de ejecución del programa (CPU): 0.000172 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000268 segundos
-------------------
Proceso 6 de 8
Tiempo de ejecución del programa (CPU): 0.000166 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000285 segundos
-------------------
Proceso 0 de 8
Tiempo de ejecución del programa (CPU): 0.000296 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000324 segundos
-------------------
Proceso 2 de 8
Tiempo de ejecución del programa (CPU): 0.000197 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000298 segundos
-------------------
Proceso 4 de 8
Tiempo de ejecución del programa (CPU): 0.000180 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000309 segundos
-------------------
Proceso 1 de 8
Tiempo de ejecución del programa (CPU): 0.000183 segundos
Tiempo de ejecución del programa (MPI_Wtime): 0.000341 segundos
```

Dando un promedio de tiempos de:

```
Tiempo promedio de ejecución del programa (CPU): 0.00019 segundos
Tiempo promedio de ejecución del programa (MPI_Wtime): 0.00029 segundos
```

### Como ejecutar

Para ejecutar el programa se necesita un compilador del lenguaje C instalado en la máquina y la librería de `OpenMPI`, y seguir los siguientes pasos:

1. Compilar el código

   ```bash
   mpicc openmpi.c -o openmpi
   ```

2. Ejecutar el código

   ```bash
   mpirun -np X ./openmpi
   ```

   Donde `X` es el número de procesos que se van a lanzar. En mi caso, usé `X = 8` porque poseo un procesador con `8 cores`.
