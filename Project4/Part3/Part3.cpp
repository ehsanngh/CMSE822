#include <iostream>
using namespace std;
#include "mpi.h"
#include "omp.h"
int main(int argc, char *argv[])
{   
    int num_threads = stoi(argv[1]);
    omp_set_num_threads(num_threads);
    const long int N = stoi(argv[2]);
    float Array1[N*N];
    float Array2[N*N];
    float answer[N*N] = { 0 };
    float answer_serial[N*N] = { 0 };

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            Array1[i*N+j] = (float)rand() / RAND_MAX * 10;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            Array2[i*N+j] = (float)rand() / RAND_MAX * 10;
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Start measuring time
    double t1, t2;
    t1 = MPI_Wtime();

    int rows_per_rank = N / size;
    float input_buffer[rows_per_rank * N];
    float answer_local[rows_per_rank * N] = { 0 };
    MPI_Scatter(Array1, rows_per_rank * N, MPI_FLOAT, input_buffer, rows_per_rank * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    #pragma omp parallel for
    for (int i = 0; i < rows_per_rank; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                answer_local[i*N+j] += input_buffer[i*N+k] * Array2[k*N+j];
            }
        }
    }
    MPI_Gather(answer_local, rows_per_rank * N, MPI_FLOAT, answer, rows_per_rank * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Stop measuring time and calculate the elapsed time
    t2 = MPI_Wtime() - t1;
    MPI_Finalize();
    //delete[] input_buffer;
    double mtotal_ops  = 2 * N * N * N * 1e-6;
    double mflops = (mtotal_ops / t2);
    
    if (rank == 0) {
        printf("N = %d \n", N);
        printf("Time measured: %.5f seconds.\n", t2);
        printf("Perfomance: %.5f MFLOPS \n", mflops);
        num_threads = omp_get_max_threads();
        cout << "Number of OpenMP threads: " << num_threads << std::endl;
    //     for (int i = 0; i < N; i++) {
    //         for (int k = 0; k < N; k++) {
    //             for (int j = 0; j < N; j++) {
    //                 answer_serial[i*N+j] += Array1[i*N+k] * Array2[k*N+j];
    //             }
    //         }
    //     }

    //     double error = 0.;
    //     for (int i = 0; i < N; i++)
    //     {
    //         for (int j = 0; j < N; j++){
    //             error += answer[i*N+j] - answer_serial[i*N+j];
    //         }
    //     }
    //     cout << "error = " << error << endl;

    }
    
    return 0;
}
