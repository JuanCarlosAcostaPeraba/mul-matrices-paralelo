/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un multiplicador secuencial de matrices

// Inclusiones
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constantes
#define N 2
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
	// Empezar contador de tiempo
	clock_t start = clock();

	// Declarar variables
	int **matriz_a, **matriz_b, **matriz_c;
	int i, j;

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

	// Imprimir tiempo de ejecución
	printf("Tiempo de ejecución: %f segundos\n", ((double) clock() - start) / CLOCKS_PER_SEC);

	// Fin del programa
	return 0;
}

/// Instrucciones para compilar el programa:
/// gcc -o secuencial secuencial.c
/// Instrucciones para ejecutar el programa:
/// ./secuencial