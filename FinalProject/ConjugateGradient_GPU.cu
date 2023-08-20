#include <iostream>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <string>
#include <cuda_runtime.h>
#include <omp.h>

using namespace std;

#define TOL 1e-6
#define MAX_ITER 1e+5
#define	threadsPerBlock	256
#define TILE_SIZE 16

/* ------------------ Element-wise Multipication addition ------------------- */
__global__ void ew_ma(double *c, double *a, double *b, double alpha, int n)
{   
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        c[tid] = a[tid] + alpha * b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
/* -------------------------------------------------------------------------- */

/* --------- Kernel helping to calculate dot product of two vectors --------- */
__global__ void partial_sum_device(double *c, double *a, double *b, int n) {
    // Shared memory to keep each x[i] * y[i]
    __shared__ double partial_sum[threadsPerBlock];  
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int index = threadIdx.x;

    double sum = 0;
    while (tid < n) 
    {
        sum += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    partial_sum[index] = sum;

    // synchronizing the threads in this block
    __syncthreads();

    /* Calculainge patial sum for a current block using data in shared memory
       using the tree reduction algorithm */
    int i = blockDim.x / 2;
    while (i != 0) {
        if (index < i)
        {partial_sum[index] += partial_sum[index + i];}
        __syncthreads();
        i /= 2;
    }
    //Store result of partial sum for a block in global memory
    if (index == 0)
        c[blockIdx.x] = partial_sum[0];
}
/* -------------------------------------------------------------------------- */

/* ------------------------ Wrapper for dot product ------------------------- */
double dot_prod_device(double *a, double *b, int n, int blocksPerGrid) {
    double *h_partial_sum, *d_partial_sum;
    double result = 0.;
    h_partial_sum = (double*)malloc(blocksPerGrid * sizeof(double));
    cudaMalloc((void**)&d_partial_sum, blocksPerGrid * sizeof(double));

    partial_sum_device<<<blocksPerGrid, threadsPerBlock>>>(d_partial_sum, a, b, n);
    cudaMemcpy(h_partial_sum, d_partial_sum,
               sizeof(double) * blocksPerGrid, cudaMemcpyDeviceToHost);
    for (int i = 0; i<blocksPerGrid; i++) 
    {
        result += h_partial_sum[i];
    }
    return result;
}
/* -------------------------------------------------------------------------- */

/* --------------------- Matrix vector multipication ------------------------ */
__global__ void mat_vec_mul_device(double *Ax, double *A, double *x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += A[tid * n + j] * x[j];
        }
        Ax[tid] = sum;
        tid += blockDim.x * gridDim.x;
    }
}

/* -------------------------------------------------------------------------- */

/* ------------- Kernel implementing the matrix multiplication -------------- */
// A * transpose(A) is calculated to create a Positive Definite Matrix
__global__ void tile_mmm_device(double* C, double* A, double* B, int n) {
    double CValue = 0;

    int Row = blockIdx.y*TILE_SIZE + threadIdx.y;
    int Col = blockIdx.x*TILE_SIZE + threadIdx.x;

    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    for (int k = 0; k < (TILE_SIZE + n - 1)/TILE_SIZE; k++) {

         if (k*TILE_SIZE + threadIdx.x < n && Row < n)
             As[threadIdx.y][threadIdx.x] = A[Row*n + k*TILE_SIZE + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_SIZE + threadIdx.y < n && Col < n)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_SIZE + threadIdx.y)*n + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int i = 0; i < TILE_SIZE; i++)
             CValue += As[threadIdx.y][i] * Bs[i][threadIdx.x];

         __syncthreads();
    }

    if (Row < n && Col < n)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*n) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}
/* -------------------------------------------------------------------------- */

void CG_device(double *x_d, double *A_d, double *b_d, int n, int blocksPerGrid) {
    double *r_d;
    double *p_d;
    double *Ap_d, *Ax_d;;

    
    cudaMalloc((void**)&r_d, sizeof(double) * n);
    cudaMalloc((void**)&p_d, sizeof(double) * n);
    cudaMalloc((void**)&Ap_d, sizeof(double) * n);
    cudaMalloc((void**)&Ax_d, sizeof(double) * n);


    mat_vec_mul_device<<<blocksPerGrid,threadsPerBlock>>>(Ax_d, A_d, x_d, n);

    // Calculation of the initial residual r_0
    ew_ma<<<blocksPerGrid,threadsPerBlock>>>(r_d, b_d, Ax_d, -1, n);

    //  Initial searching direction p_0 = r_0
    cudaMemcpy(p_d, r_d, n * sizeof(double), cudaMemcpyDeviceToDevice);  

    int k = 0;
    double norm_error = sqrt(dot_prod_device(r_d, r_d, n, blocksPerGrid));
    
    auto start = chrono::high_resolution_clock::now();
    while (norm_error > TOL && k < MAX_ITER) {
        // Calculating the step size alpha
        mat_vec_mul_device<<<blocksPerGrid,threadsPerBlock>>>(Ap_d, A_d, p_d, n); 
        double alpha = dot_prod_device(r_d, p_d, n, blocksPerGrid)
                     / dot_prod_device(p_d, Ap_d, n, blocksPerGrid);
        
        // Updating the solution vector
        ew_ma<<<blocksPerGrid,threadsPerBlock>>>(x_d, x_d, p_d, alpha, n);

        // Updating the residual vector
        mat_vec_mul_device<<<blocksPerGrid,threadsPerBlock>>>(Ax_d, A_d, x_d, n); 
        ew_ma<<<blocksPerGrid,threadsPerBlock>>>(r_d, b_d, Ax_d, -1, n);
        norm_error = sqrt(dot_prod_device(r_d, r_d, n, blocksPerGrid));
            
        // Calculating the step size beta
        double beta = - dot_prod_device(r_d, Ap_d, n, blocksPerGrid)
                      / dot_prod_device(p_d, Ap_d, n, blocksPerGrid);
    
        // Updating the search direction vector
        ew_ma<<<blocksPerGrid,threadsPerBlock>>>(p_d, r_d, p_d, beta, n);

        k++;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end - start;
    
    cout << " ----------------- Results from GPU ------------------- " << endl;
    cout << "Number of Iterations: " << k << endl;
    cout << "Norm of Residuals: " << norm_error << endl;
    cout << "Elapsed time: " << elapsed_time.count() << " seconds\n";
    cout << " ------------------------------------------------------ " << endl;
    cout << endl;

    cudaFree(Ax_d); 
    cudaFree(Ap_d); 
    cudaFree(p_d);
    cudaFree(r_d);
}
/* -------------------------------------------------------------------------- */

/* ------------------------------- CPU Solver ------------------------------- */
double dot_prod_host(double *x, double *y, int n) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
}

double norm(double* x, int n) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += x[i] * x[i];
    }
    return sqrt(result);
}

void mat_vec_mul(double *Ax, double *A, double *x, int n) {
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i*n+j] * x[j];
        }
        Ax[i] = sum;
    }
    
}

void pos_def_matrix_generator(double *AAT, double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n+k] * A[j*n+k];
            }
            AAT[i*n+j] = sum;
        }
    }
}

void CG_host(double *x, double *A, double *b, int n) {
    double *Ax;
    Ax = (double*)malloc(sizeof(double) * n);
    mat_vec_mul(Ax, A, x, n);
    double *r, *p, *Ap;
    r = (double*)malloc(sizeof(double) * n);
    p = (double*)malloc(sizeof(double) * n);
    Ap = (double*)malloc(sizeof(double) * n);

    // Calculation of the initial residual r_0, p_0
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ax[i];
        p[i] = r[i];
    }
    

    int k = 0;

    auto start = chrono::high_resolution_clock::now();
    while (norm(r, n) > TOL && k < MAX_ITER) {
        // calculate the step size alpha
        mat_vec_mul(Ap, A, p, n);
        
        double alpha = dot_prod_host(r, p, n) / dot_prod_host(p, Ap, n);
        // update the solution vector

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
        }

        // update the residual vector
        mat_vec_mul(Ax, A, x, n);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - Ax[i];
        }
        

        // calculate the step size beta
        double beta = - dot_prod_host(r, Ap, n) / dot_prod_host(p, Ap, n);

        // update the search direction vector

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        k++;
    }
    
    free(Ax); 
    free(Ap); 
    free(p); 
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end - start;
    cout << " ----------------- Results from CPU ------------------- " << endl;
    cout << "Number of Iterations: " << k << endl;
    cout << "Norm of Residuals: " << norm(r, n) << endl;
    cout << "Elapsed time: " << elapsed_time.count() << " seconds\n";
    cout << " ------------------------------------------------------ " << endl;
    free(r);

}

/* -------------------------------------------------------------------------- */


int main(int argc, char* argv[]) {
    int num_threads = stoi(argv[3]);
    omp_set_num_threads(num_threads);
    const int N = stoi(argv[1]);
    cout << "Matrix size = " << N << endl;
    string device_type = argv[2];
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    cout << "device_count = " << device_count << endl;
    cout << "Number of OMP Threads: " << num_threads << endl;
    cout << endl;

    /*------------------------ Creating x and b ------------------------------*/
    double *x_h;
    double *b_h;

    x_h = (double*)malloc(sizeof(double) * N);
    b_h = (double*)malloc(sizeof(double) * N);

    for (int i = 0; i < N; i++) {
        x_h[i] = (double)rand() / RAND_MAX * 10;
        b_h[i] = (double)rand() / RAND_MAX * 10;
    }
    
    /* ---------------------------------------------------------------------- */

    /*------------ Creating the Positive Definite Matrix of A ----------------*/
    double *initial_A_h, *initial_A_h_T, *A_h;
    double *initial_A_d, *initial_A_d_T, *A_d;

    A_h = (double*)malloc(sizeof(double) * N * N);
    initial_A_h = (double*)malloc(sizeof(double) * N * N);
    initial_A_h_T = (double*)malloc(sizeof(double) * N * N);

    // Creating a random matrix initial_A
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            initial_A_h[i*N+j] = (double)rand() / RAND_MAX * 10;
            initial_A_h_T[j*N+i] = initial_A_h[i*N+j];
        }
    }

     if (device_count == 0 || device_count > 16) {
        cout << "No GPU available" << endl;
        pos_def_matrix_generator(A_h, initial_A_h, N);
        free(initial_A_h);
        free(initial_A_h_T);
     }
     else {
        dim3 blockDim(16, 16);
        dim3 gridDim(ceil(N/(double)TILE_SIZE),
                     ceil(N/(double)TILE_SIZE));
        
        cudaMalloc((void**)&A_d, sizeof(double) * N * N);
        cudaMalloc((void**)&initial_A_d, sizeof(double) * N * N);
        cudaMalloc((void**)&initial_A_d_T, sizeof(double) * N * N);
        cudaMemcpy(initial_A_d, initial_A_h, sizeof(double) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(initial_A_d_T, initial_A_h_T, sizeof(double) * N * N, cudaMemcpyHostToDevice);
        free(initial_A_h);
        free(initial_A_h_T);
        tile_mmm_device<<<gridDim,blockDim>>>(A_d, initial_A_d, initial_A_d_T, N); 
        cudaMemcpy(A_h, A_d, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
        cudaFree(initial_A_d);
        cudaFree(initial_A_d_T);
     }
    
    
    /*------------------------------------------------------------------------*/
    
    if (device_type == "CPU") {
        CG_host(x_h, A_h, b_h, N);
    }
    else if (device_type == "GPU") {
        double *x_d, *b_d;
        cudaMalloc((void**)&x_d, sizeof(double) * N);
        cudaMalloc((void**)&b_d, sizeof(double) * N);
        cudaMemcpy(x_d, x_h, sizeof(double) * N, cudaMemcpyHostToDevice);     
        cudaMemcpy(b_d, b_h, sizeof(double) * N, cudaMemcpyHostToDevice);
        int number_of_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid = (32 < number_of_blocks ? 32 : number_of_blocks);
        CG_device(x_d, A_d, b_d, N, blocksPerGrid);

        cudaFree(A_d);
        cudaFree(b_d);
        cudaFree(x_d);
    }
    else if (device_type == "both") {
        double *x_d, *b_d;
        cudaMalloc((void**)&x_d, sizeof(double) * N);
        cudaMalloc((void**)&b_d, sizeof(double) * N);
        cudaMemcpy(x_d, x_h, sizeof(double) * N, cudaMemcpyHostToDevice);     
        cudaMemcpy(b_d, b_h, sizeof(double) * N, cudaMemcpyHostToDevice);
        int number_of_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGrid = (32 < number_of_blocks ? 32 : number_of_blocks);
        CG_device(x_d, A_d, b_d, N, blocksPerGrid);

        CG_host(x_h, A_h, b_h, N);
        double *x_d_on_h;
        x_d_on_h = (double*)malloc(sizeof(double) * N);
        cudaMemcpy(x_d_on_h, x_d, sizeof(double) * N, cudaMemcpyDeviceToHost);

        double MSE = 0.;
        for(int i = 0; i < N; i++){
            MSE += (x_h[i] - x_d_on_h[i]) * (x_h[i] - x_d_on_h[i]);
        }
        MSE = sqrt(MSE) / N;

        cout << "MSE: " << MSE << endl;
        free(x_d_on_h);
        cudaFree(A_d);
        cudaFree(b_d);
        cudaFree(x_d);

    }
    else {
    cerr << "Invalid device type: " << device_type << endl;
    return 1;
    }

    free(x_h);
    free(b_h);
    free(A_h);

    return 0;
}
