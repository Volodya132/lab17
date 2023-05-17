
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>
#include <Windows.h>
#include <iomanip>

using namespace std;

__global__ void compute(double* a, double* b, double* c, double* d, double* res)
{
    int idx = threadIdx.x;
    res[idx] = (a[idx] + b[idx] + c[idx] - d[idx])/2.;
}

void randomGenerateArr(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

void printArr(double* arr, int size) {
    cout << fixed;
    cout.precision(2);

    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}
int main()
{
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    srand(time(NULL));
    const int size = 7;
    double cpu_arr1[size];
    double cpu_arr2[size];
    double cpu_arr3[size];
    double cpu_arr4[size];
    double cpu_res[size];

    double* gpu_arr1, * gpu_arr2, * gpu_arr3, *gpu_arr4, *gpu_res;
    randomGenerateArr(cpu_arr1, size);
    randomGenerateArr(cpu_arr2, size);
    randomGenerateArr(cpu_arr3, size);
    randomGenerateArr(cpu_arr4, size);

    cout << "a: " << endl;
    printArr(cpu_arr1, size);

    cout << "b: " << endl;
    printArr(cpu_arr2, size);

    cout << "c: " << endl;
    printArr(cpu_arr3, size);

    cout << "d: " << endl;
    printArr(cpu_arr4, size);

    cudaMalloc((void**)&gpu_arr1, sizeof(double)*size);
    cudaMalloc((void**)&gpu_arr2, sizeof(double) * size);
    cudaMalloc((void**)&gpu_arr3, sizeof(double) * size);
    cudaMalloc((void**)&gpu_arr4, sizeof(double) * size);
    cudaMalloc((void**)&gpu_res, sizeof(double) * size);

    cudaMemcpy(gpu_arr1, cpu_arr1, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_arr2, cpu_arr2, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_arr3, cpu_arr3, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_arr4, cpu_arr4, sizeof(double) * size, cudaMemcpyHostToDevice);

    compute << <1, size >> > (gpu_arr1, gpu_arr2, gpu_arr3, gpu_arr4, gpu_res);

    cudaMemcpy(cpu_res, gpu_res, sizeof(double) * size, cudaMemcpyDeviceToHost);
    
    cout << "res: " << endl;
    printArr(cpu_res, size);


    cudaFree(gpu_arr1);
    cudaFree(gpu_arr2);
    cudaFree(gpu_arr3);
    cudaFree(gpu_arr4);
    cudaFree(gpu_res);



    return 0;
}

