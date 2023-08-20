#include <iostream>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <omp.h>
using namespace std;

#define TOL 1e-6
#define MAX_ITER 1e+4
/* ------------------------------ Host Solver ------------------------------- */
double dot_product(double *x, double *y, int n) {
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


int main(int argc, char* argv[]) {
    const int N = stoi(argv[1]);
    int num_threads = stoi(argv[2]);
    omp_set_num_threads(num_threads);

    double *initial_A, *A;

    initial_A = (double*)malloc(sizeof(double) * N * N);
    A = (double*)malloc(sizeof(double) * N * N);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            initial_A[i*N+j] = (double)rand() / RAND_MAX * 10;
    }

    pos_def_matrix_generator(A, initial_A, N);
    free(initial_A);
    
    double *x;
    x = (double*)malloc(sizeof(double) * N);

    double *b;
    b = (double*)malloc(sizeof(double) * N);

    // Initialization of the solution vector x_0
    for (int i = 0; i < N; i++) {
        x[i] = (double)rand() / RAND_MAX * 10;
        b[i] = (double)rand() / RAND_MAX * 10;
    }
    
    
    cout << "Matrix size:" << N << endl;
    double *Ax;
    Ax = (double*)malloc(sizeof(double) * N);
    mat_vec_mul(Ax, A, x, N);
    double *r, *p, *Ap;
    r = (double*)malloc(sizeof(double) * N);
    p = (double*)malloc(sizeof(double) * N);
    Ap = (double*)malloc(sizeof(double) * N);

    // Calculation of the initial residual r_0, p_0
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        r[i] = b[i] - Ax[i];
        p[i] = r[i];
    }
    

    int k = 0;

    auto start = chrono::high_resolution_clock::now();
    while (norm(r, N) > TOL && k < MAX_ITER) {
        // calculate the step size alpha
        mat_vec_mul(Ap, A, p, N);
        double alpha = dot_product(r, p, N) / dot_product(p, Ap, N);
        // update the solution vector
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
        }

        // update the residual vector
        mat_vec_mul(Ax, A, x, N);
        #pragma omp parallel for
        for (int i = 0; i <N; i++) {
            r[i] = b[i] - Ax[i];
        }
        

        // calculate the step size beta
        double beta = - dot_product(r, Ap, N) / dot_product(p, Ap, N);

        // update the search direction vector
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
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
    cout << "Norm of Residuals: " << norm(r, N) << endl;
    free(r);
    cout << "Elapsed time: " << elapsed_time.count() << " seconds\n";
    cout << " ------------------------------------------------------ " << endl;
    free(A);
    free(b);
    free(x);
    return 0;
}
