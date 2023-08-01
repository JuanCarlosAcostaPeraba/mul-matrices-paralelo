/// Trabajo Práctico 3. Programación paralela de la multiplicación de matrices

/// Implementación en un coprocesador de tipo GPU usando CUDA

// Inclusiones
#include <stdio.h>
#include <stdlib.h>

// Constantes
#define N 1000 // Tamaño de las matrices

// Funciones
void multiplicar_matrices(int **matriz_a, int **matriz_b, int **matriz_c) {
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

	// Empezar contador de tiempo CUDA

	// Reserva de memoria para las matrices

	// Inicializar matrices

	// Multiplicar matrices

	// Imprimir matrices

	// Liberar memoria de las matrices

	gettimeofday(&fin, NULL);

	// Imprimir tiempo de ejecución CUDA
	
	// Finalizar programa normalmente
	return 0;
}

/// Instrucciones para compilar el programa:
/// 
/// Instrucciones para ejecutar el programa:
/// 