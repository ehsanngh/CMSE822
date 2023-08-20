#include <iostream>
using namespace std;
#include <stdio.h>
#include <sys/time.h>
int main()
{
    const long int r1 = 10000;
    const int c1 = r1;
    const int r2 = c1;
    const int c2 = r2;
    float Array1[r1][c1];
    float Array2[r2][c2];
    float answer[r1][c2] = { 0 };
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++)
            Array1[i][j] = (float)rand() / RAND_MAX * 10;
    }
    for (int i = 0; i < r2; i++) {
        for (int j = 0; j < c2; j++)
            Array2[i][j] = (float)rand() / RAND_MAX * 10;
    }
    /*for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++)
            cout << Array1[i][j] << ” “;
        cout << endl;
    }
    cout << ” -------------------- “;
    cout << endl;
    for (int i = 0; i < r2; i++) {
        for (int j = 0; j < c2; j++)
            cout << Array2[i][j] << ” “;
        cout << endl;
    }
    cout << ” -------------------- “;
    cout << endl;*/
    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    for (int i = 0; i < r1; i++) {
        for (int k = 0; k < c2; k++) {
            for (int j = 0; j < c1; j++) {
                answer[i][k] += Array1[i][j] * Array2[j][k];
            }
        }
    }
    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    double seconds = end.tv_sec - begin.tv_sec;
    double microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    double mtotal_ops  = 2 * r1 * r1 * r1*1e-6;
    double mflops = (mtotal_ops / elapsed);
    printf("N = %d \n", r1);
    printf("Time measured: %.5f seconds.\n", elapsed);
    printf("Perfomance: %.5f MFLOPS \n", mflops);
    /*cout << ” -------------------------------- “;
    cout << endl;
    cout << “answer = “;
    cout << endl;
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++)
            cout << answer[i][j] << ” “;
        cout << endl;
    }*/
    return 0;
}
