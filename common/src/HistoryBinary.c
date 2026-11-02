#include <stdio.h>
#include <stdint.h>
#include "include/Index.h" // IDX, isRecorded
#include "include/HistoryBinary.h"

int writeHistoryBinary(const char* filename, const double* u, int N, int T, double xL, double xR, double dt, int every)
{
    FILE* fp = fopen(filename, "wb"); // Open the file or create it in case it does not exist
    if (fp == NULL) {
        perror(filename);
        return 1;
    }

    // Header
    int64_t nCells    = N + 1;
    int64_t nRecords  = 0;
    for (int n = 0; n < T; ++n) nRecords += isRecorded(n, T, every);
    double  domain[2] = { xL, xR };

    fwrite(&nCells,   sizeof(int64_t), 1, fp);
    fwrite(&nRecords, sizeof(int64_t), 1, fp);
    fwrite(domain,    sizeof(double),  2, fp);

    // Records: time, then the cells of that time step (contiguous in the history, ghost cells excluded)
    for (int n = 0; n < T; ++n) {
        if (!isRecorded(n, T, every)) continue;
        double t = n*dt;
        fwrite(&t,               sizeof(double), 1,             fp);
        fwrite(u + IDX(n, 0, N), sizeof(double), (size_t)(N+1), fp);
    }

    int failed = ferror(fp);
    if (fclose(fp) != 0 || failed) {
        perror(filename);
        return 1;
    }
    return 0;
}
