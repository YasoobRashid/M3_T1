#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 750;

void generateMatrix(vector<int> &matrix) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = rand() % 10;
    }
}

void multiplyChunk(const vector<int> &A_chunk, const vector<int> &B, vector<int> &C_chunk, int rowsPerProc) {
    for (int i = 0; i < rowsPerProc; i++) {
        for (int j = 0; j < N; j++) {
            C_chunk[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C_chunk[i * N + j] += A_chunk[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsPerProc = N / size;

    vector<int> A, B(N * N), C;

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        generateMatrix(A);
        generateMatrix(B);
    }

    vector<int> A_chunk(rowsPerProc * N);
    vector<int> C_chunk(rowsPerProc * N);

    MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(A.data(), rowsPerProc * N, MPI_INT, A_chunk.data(), rowsPerProc * N, MPI_INT, 0, MPI_COMM_WORLD);

    auto start = high_resolution_clock::now();

    multiplyChunk(A_chunk, B, C_chunk, rowsPerProc);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    MPI_Gather(C_chunk.data(), rowsPerProc * N, MPI_INT, C.data(), rowsPerProc * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "MPI Execution Time: " << duration.count() << " ms" << endl;
    }

    MPI_Finalize();
    return 0;
}
