# Programación paralela de la multiplicación de matrices

![GitHub](https://img.shields.io/github/license/juancarlosacostaperaba/mul-matrices-paralelo)

## Programa Secuencial

El programa secuencial se encuentra en el archivo [secuencial.c](./secuencial.c) y se compila con el comando:

```bash
# gcc <nombre del código C>.c -o <nombre del ejecutable>
gcc secuencial.c -o secuencial
```

Para ejecutar el programa se debe ejecutar el comando:

```bash
./secuencial
```

## Programa Paralelo con OpenMP

El programa paralelo se encuentra en el archivo [openmp.c](./openmp.c) y se compila con el comando:

```bash
# gcc <nombre del código C>.c -o <nombre del ejecutable> -fopenmp
gcc openmp.c -o openmp -fopenmp
```

Para ejecutar el programa se debe ejecutar el comando:

```bash
./openmp
```

## Programa Paralelo con OpenMPI

El programa paralelo se encuentra en el archivo [openmpi.c](./openmpi.c) y se compila con el comando:

```bash
# mpicc <nombre del código C>.c -o <nombre del ejecutable>
mpicc openmpi.c -o openmpi
```

Para ejecutar el programa se debe ejecutar el comando:

```bash
# mpirun -np <número de procesos> ./<nombre del ejecutable>
mpirun -np 8 ./openmpi
```

## Programa Paralelo con CUDA

El programa paralelo se encuentra en el archivo [cuda.cu](./cuda.cu) y se compila y ejecuta con el IDE _[Microsoft Visual Studio](https://visualstudio.microsoft.com/es/downloads/)_ y con la [librería de CUDA](https://developer.nvidia.com/cuda-zone) desarrollada por NVIDIA.

Para compilar y ejecutar hay que seguir los siguientes pasos:

1. Crear un nuevo proyecto de CUDA en Visual Studio y copiar el código del archivo `cuda.cu` en el archivo `kernel.cu` que se crea por defecto.

2. Compilar y ejecutar el proyecto con el IDE.
   ![Botón para compilar y ejecutar el código CUDA en el IDE Microsoft Visual Studio](./boton-compilar-cuda.png)
