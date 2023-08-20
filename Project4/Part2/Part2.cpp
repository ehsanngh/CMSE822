#include "mpi.h"
#include <iostream>
//#include <sstream>
using namespace std;
#include "omp.h"

int main(int argc, char *argv[]) 
{   
    int num_threads = stoi(argv[1]);
    omp_set_num_threads(num_threads);
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    #pragma omp parallel
    {
        // stringstream ss;
        cout << "Hello, World! " << "from rank: " << rank << " and from thread: "
        << omp_get_thread_num() << endl;
        // cout << ss.str();
    }

    MPI_Finalize();
    return 0;
}
