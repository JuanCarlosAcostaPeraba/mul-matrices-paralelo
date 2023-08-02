# Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

## Objetivo

Implementar el algoritmo de multiplicación de matrices con números en coma flotante en las librerías paralelas OpenMP, OpenMPI y CUDA utilizando un ordenador sobremesa.

## Tareas a realizar

- [x] Implementación en un multiprocesador usando OpenMP
- [x] Implementación en un multicomputador usando OpenMPI
- [x] Implementación en un coprocesador de tipo GPU usando CUDA
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

```bash
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

```bash
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
```

### Explicación

En este programa se realiza la multiplicación de dos matrices cuadradas de tamaño `N = 8` de forma paralela tratando cada `core` del procesador como un ordenador independiente. Por este motivo, el tamaño de las matrices no pueden ser mayores, ya que no dispongo de más núcleos.

El tiempo individual que ha tardado este programa en ejecutarse en mi ordenador ha sido de:

```bash
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

```bash
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

## Implementación en un coprocesador de tipo GPU usando CUDA

### Código

```c
/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <time.h>

// Definiciones
#define N 2
#define BLOCK_SIZE 2 // Debe ser igual a N
#define CLOCKS_PER_SEC 1000000

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

	// Empezar contador de tiempo
	clock_t start = clock(); // CPU

	// Reserva de memoria en CPU
	a_cpu = (int *)malloc(size);
	b_cpu = (int *)malloc(size);
	c_cpu = (int *)malloc(size);

	// Inicializar matrices
	srand(time(NULL)); // Semilla aleatoria
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

	memset(c_cpu, 0, size);

	// Definir bloques e hilos
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	// Lanzar kernel
	multiplicar_matrices<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, N);

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error de CUDA: %s\n", cudaGetErrorString(error));
	}

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

	// Imprimir tiempo de ejecución
	printf("\n-------------------\n");
	printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double) clock() - start) / CLOCKS_PER_SEC);

	return 0;
}
```

### Explicación

En este programa se realiza la multiplicación de dos matrices cuadradas de tamaño `N = 1024` en una tarjera gráfica Nvidia con los núcleos CUDA de la misma, enviando los datos en bloques de datos de `32`. Esto es así porque tanto el tamaño de las matrices como el tamaño de los bloques deben ser exponentes de `2`.

El tiempo que ha tardado este programa en ejecutarse en mi ordenador ha sido de:

```bash
-------------------
Tiempo de ejecución del programa (CPU): 0.057555 segundos
```

### Como ejecutar

Para ejecutar este código hay que seguir los siguientes pasos:

1. Poseer un onrdenador con una gráfica Nvidia con Núcleos CUDA, y que esta gráfica sea de una gama media alta, ya que si es media baja, lo más probable es que o no posea los núcleos CUDA o que usen versiones muy antiguas que no permitirán realizar los cálculos.
2. Instalar Microsoft Visual Studio, con la partición de _Crear aplicaciones de escritorio con C++_ para obtener así el compilador de C/C++.
3. Instalar el kit de desarrollo de CUDA (_CUDA Toolkit_), con la instalación personalizada, ya que hay que marcar la opción de _Visual Studio Integration_.
4. Crear un nuevo proyecto en CUDA y pegar el código en el archivo `kernel.cu`.
5. Finalmente, ejecutar el código con el ide, dándole al botón remarcado. Este compilará el codigo la primera vez, y lo ejecutará de seguido.

   ![Botón del play](https://file.notion.so/f/s/4345d100-04c9-475d-944f-807d1c36d92c/Untitled.png?id=af52c0aa-1416-4446-ad29-f4bb344f17b6&table=block&spaceId=468e4f0a-eef2-4192-a273-959b9a958a93&expirationTimestamp=1691085600000&signature=npNxcTxDE0MazRR9JVwA83c7qnwfvoUwDG6GgyVNU2w&downloadName=Untitled.png)

6. Posteriormente, si se queiren hacer cambios, se le da al mismo botón, ya que por cada cambio, el código se compila de nuevo.
