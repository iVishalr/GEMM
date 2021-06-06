#include <gemm.h>

#define A(i,j) a[(j)*lda+(i)]
#define B(i,j) b[(j)*ldb+(i)]
#define C(i,j) c[(j)*ldc+(i)]

#define mc 256
#define kc 128
#define nb 1000

#define min(i,j) ((i)<(j) ? (i):(j))

typedef union{
    __m128d v;
    double d[2];
}v2df_t;

void serial_init(int m, int n, double * a, int lda){
    int count = 1;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++)
            A(i,j) = count++;
    }
}

void random_init(int m, int n, double * a, int lda){
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++)
            A(i,j) = 2.0 * drand48() - 1.0;
    }
}

void display(double * matrix, int m, int n){
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            printf("%f ",matrix[j*m+i]);
        }
        printf("\n");
    }
    return;
}

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){

    int p;
  v2df_t
    c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
    c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,
    a_0p_a_1p_vreg,
    a_2p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 

  c_00_c_10_vreg.v = _mm_setzero_pd();   
  c_01_c_11_vreg.v = _mm_setzero_pd();
  c_02_c_12_vreg.v = _mm_setzero_pd(); 
  c_03_c_13_vreg.v = _mm_setzero_pd(); 
  c_20_c_30_vreg.v = _mm_setzero_pd();   
  c_21_c_31_vreg.v = _mm_setzero_pd();  
  c_22_c_32_vreg.v = _mm_setzero_pd();   
  c_23_c_33_vreg.v = _mm_setzero_pd(); 

  for ( p=0; p<k; p++ ){
    a_0p_a_1p_vreg.v = _mm_load_pd( (double *) a );
    a_2p_a_3p_vreg.v = _mm_load_pd( (double *) ( a+2 ) );
    a += 4;

    b_p0_vreg.v = _mm_loaddup_pd( (double *) b );       /* load and duplicate */
    b_p1_vreg.v = _mm_loaddup_pd( (double *) (b+1) );   /* load and duplicate */
    b_p2_vreg.v = _mm_loaddup_pd( (double *) (b+2) );   /* load and duplicate */
    b_p3_vreg.v = _mm_loaddup_pd( (double *) (b+3) );   /* load and duplicate */

    b += 4;

    /* First row and second rows */
    c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
    c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
    c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
    c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

    /* Third and fourth rows */
    c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
    c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
    c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
    c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
  }

  C( 0, 0 ) += c_00_c_10_vreg.d[0];  C( 0, 1 ) += c_01_c_11_vreg.d[0];  
  C( 0, 2 ) += c_02_c_12_vreg.d[0];  C( 0, 3 ) += c_03_c_13_vreg.d[0]; 

  C( 1, 0 ) += c_00_c_10_vreg.d[1];  C( 1, 1 ) += c_01_c_11_vreg.d[1];  
  C( 1, 2 ) += c_02_c_12_vreg.d[1];  C( 1, 3 ) += c_03_c_13_vreg.d[1]; 

  C( 2, 0 ) += c_20_c_30_vreg.d[0];  C( 2, 1 ) += c_21_c_31_vreg.d[0];  
  C( 2, 2 ) += c_22_c_32_vreg.d[0];  C( 2, 3 ) += c_23_c_33_vreg.d[0]; 

  C( 3, 0 ) += c_20_c_30_vreg.d[1];  C( 3, 1 ) += c_21_c_31_vreg.d[1];  
  C( 3, 2 ) += c_22_c_32_vreg.d[1];  C( 3, 3 ) += c_23_c_33_vreg.d[1]; 
}

void PackMatrixA(int k, double *a, int lda, double * a_to){
    int j;

    for( j=0; j<k; j++){  /* loop over columns of A */
        double 
        *a_ij_pntr = &A( 0, j );

        *a_to     = *a_ij_pntr;
        *(a_to+1) = *(a_ij_pntr+1);
        *(a_to+2) = *(a_ij_pntr+2);
        *(a_to+3) = *(a_ij_pntr+3);

        a_to += 4;
    }
}

void PackMatrixB(int k, double *b, int ldb, double *b_to){
    int i;
    double 
        *b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
        *b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

    for( i=0; i<k; i++){  /* loop over rows of B */
        *b_to++ = *b_i0_pntr++;
        *b_to++ = *b_i1_pntr++;
        *b_to++ = *b_i2_pntr++;
        *b_to++ = *b_i3_pntr++;
    }
}

void InnerKernel(int m, int n, int k, double * a, int lda, double *b, int ldb, double *c, int ldc,int first_time){
    int i, j;
    //   double 
    //     packedA[ m * k ];
    //   static double 
    //     packedB[ kc*nb ];    /* Note: using a static buffer is not thread safe... */
    double * packedA = (double*)calloc(m*k,sizeof(double));
    double * packedB = (double*)calloc(kc*nb,sizeof(double));

    for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
        if ( first_time )
            PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
        for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
            /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
            one routine (four inner products) */
                if ( j == 0 ) 
                PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
            AddDot4x4( k, &packedA[ i*k ], 4, &packedB[ j*k ], k, &C( i,j ), ldc );
        }
    }
}

void matmul(int m, int n, int k, double * a, int lda, double * b, int ldb, double * c, int ldc){
	/*
	Computes the matrix multiplication of A and B and stores in C.
	C = A*B + C
	Arguments
	---------
		m,n,k : Specifies matrix dimensions
		a : pointer to first matrix
		b : pointer to second matrix
		c : pointer to the resultant matrix
		lda : leading dimension of matrix a
		ldb : leading dimension of matrix b
		ldc : leading dimension of matrix c

	Return
	------
		None
	*/

	if(a==NULL || b==NULL || c==NULL){
		printf("Argument Error : One of the input arguments to matmul() was NULL\n");
		return;
	}

    int i, p, pb, ib;

    /* This time, we compute a mc x n block of C by a call to the InnerKernel */

    for ( p=0; p<k; p+=kc ){
        pb = min( k-p, kc );
        for ( i=0; i<m; i+=mc ){
        ib = min( m-i, mc );
        InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc, i==0 );
        }
    }
	return;
}


int main(){

    int m = 2000;
    int n = 2000;
    int k = 2000;

    double * A = (double*)calloc(m*k,sizeof(double));
    double * B = (double*)calloc(k*n,sizeof(double));
    double * C = (double*)calloc(m*n,sizeof(double));

    struct timeval start,finish;
    double gflops = 2.0 * m*n*k * 1.0e-09;
    srand((unsigned)time(NULL));

    if(A==NULL || B==NULL || C==NULL){
        printf("Out of Memory!\n");
        exit(EXIT_FAILURE);
    }

    random_init(m,k,A,m);
    random_init(k,n,B,k);

    gettimeofday(&start, NULL);
    matmul(m,n,k,A,m,B,k,C,m);
    gettimeofday(&finish, NULL);

    double duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;

    if(m<=8){
        printf("Matrix A : \n");
        display(A,m,k);

        printf("Matrix B : \n");
        display(B,k,n);

        printf("Dot Product Result : \n");
        display(C,m,n);
    }
    
    printf("Optimization 14 : Dot product took %f seconds GFLOPS : %f\n",duration,gflops/duration);
    return 0;
}