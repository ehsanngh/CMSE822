#include "mpi.h"
#include <cstdio>
using namespace std;


int main(int argc, char *argv[]) 
{
    MPI_Init(&argc, &argv);
    
    int nameLen = MPI_MAX_PROCESSOR_NAME;
    char procName[nameLen];
    MPI_Get_processor_name(procName, &nameLen);
    printf("Proc Name: %s\n", procName);    
    

    MPI_Finalize();
    return 0;
}
