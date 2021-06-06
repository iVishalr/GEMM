#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <mmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>


void AddDot4x4( int, double *, int, double *, int, double *, int );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void InnerKernel( int, int, int, double *, int, double *, int, double *, int, int );
void matmul( int m, int n, int k, double *a, int lda, double *b, int ldb,double *c, int ldc );