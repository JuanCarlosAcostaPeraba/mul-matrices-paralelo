/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un multiplicador secuencial de matrices

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constantes
#define N 1000
#define CLOCKS_PER_SEC 1000000

// Función para multiplicar matrices
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

// Función principal
int main(int argc, char *argv[]) {
	// Declaración de variables
	int **matriz_a, **matriz_b, **matriz_c;
	int i, j;
	srand(time(NULL)); // semilla para generar números aleatorios

	// Empezar contador de tiempo
	clock_t start = clock(); // CPU

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
	printf("\nMatriz A:\n");
	imprimir_matriz(matriz_a);
	printf("\nMatriz B:\n");
	imprimir_matriz(matriz_b);
	printf("\nMatriz C (resultado):\n");
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

	// Imprimir tiempo de ejecución
	printf("\n-------------------\n");
	printf("Tiempo de ejecución del programa (CPU): %f segundos\n", ((double) clock() - start) / CLOCKS_PER_SEC);

	// Fin del programa
	return 0;
}