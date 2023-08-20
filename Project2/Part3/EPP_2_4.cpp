#include "mpi.h"
#include <cstdio>
using namespace std;


int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);

    MPI_Comm my_comm = MPI_COMM_WORLD;
    int size, rank;

    MPI_Comm_size(my_comm, &size);
    MPI_Comm_rank(my_comm, &rank);

    printf("Process: %d out of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
