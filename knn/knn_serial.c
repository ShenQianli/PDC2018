#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE 10000

int main(void) {
    // read inputs from file
    FILE * fp = fopen("./coordinates.txt", "r");
    // n: number of points
    // dim: dimensions
    // topK: find the topK nearest
    int n, dim, topK, i, j, d;
    fscanf(fp, "%d\n%d\n%d", &n, &dim, &topK);

    // M: store the coordinates of points, size: n * dim
    // Dist: store the distance of points, size: n * n
    // Res: store the kth nearest points of a point, size: n * topK
    float *M = (float *) malloc(dim * n * sizeof(float));
    float *Dist = (float *)malloc(n * n * sizeof(float));
    int *Res = (int *)malloc(n * topK * sizeof(int));

    // read the points coordinates
    for (i = 0; i < n; ++i){
        for (j = 0; j < dim; ++j){
            fscanf(fp, "%f", M + i * dim + j);
        }
    }
    fclose(fp);
 
    clock_t start_time = clock();
    // calculate the distance
    for (i = 0; i < n; ++i){
        for (j = 0; j <= i; ++j){
            if (i == j){
                Dist[i * n + i] = 3e10;
                continue;
            } 
            float distance = 0.0;
            for (d = 0; d < dim; ++d){
                float p1 = M[dim * i + d];
                float p2 = M[dim * j + d];
                distance += (p1 - p2) * (p1 - p2);
            }
            Dist[i * n + j] = distance;
            Dist[j * n + i] = distance;
        }
    }

    // select the smallest K
    for (i = 0; i < n; ++i){
        for (j = 0; j < topK; ++j){
            float MAX = 3e10;
            int MAXidx = -1;
            for (d = 0; d < n; ++d){
                if (Dist[i * n + d] < MAX){
                    MAX = Dist[i * n + d];
                    MAXidx = d;
                }
            }
            
            Res[i * topK + j] = MAXidx;
            Dist[MAXidx + i * n] = 3e10;
        }
    }

    clock_t stop_time = clock();
    double elapsed = (double)(stop_time - start_time)* 1000.0 / CLOCKS_PER_SEC;
    printf("Total Time: %.3fms\n", elapsed);
    // Display the result to the screen
    for (i = n-10; i < n; ++i){
        printf("knn of %d: ", i);
        // PrintPoint(Dist, dim, i, n);
        for (j = 0; j < topK; ++j){
            int v = Res[i * topK + j];
            printf(" %d", v);
            // PrintPoint(Dist, dim, v, n);
        }
        printf("\n");
    }


    free(M);
    free(Dist);
    free(Res);
    return 0;
}
