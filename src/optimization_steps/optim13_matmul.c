#include <gemm.h>

#define A(i,j) a[(j)*lda+(i)]
#define B(i,j) b[(j)*ldb+(i)]
#define C(i,j) c[(j)*ldc+(i)]

#define mc 256
#define kc 128

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

    v2df_t
        c_00_c_10_vreg, c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg,
        c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg,
        a_0p_a_1p_vreg,
        a_2p_a_3p_vreg,
        b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;

    double *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;

    b_p0_ptr = &B(0,0);
    b_p1_ptr = &B(0,1);
    b_p2_ptr = &B(0,2);
    b_p3_ptr = &B(0,3);

    c_00_c_10_vreg.v = _mm_setzero_pd(); //initializes the vect register with 0s
    c_01_c_11_vreg.v = _mm_setzero_pd();
    c_02_c_12_vreg.v = _mm_setzero_pd();
    c_03_c_13_vreg.v = _mm_setzero_pd();
    c_20_c_30_vreg.v = _mm_setzero_pd();
    c_21_c_31_vreg.v = _mm_setzero_pd();
    c_22_c_32_vreg.v = _mm_setzero_pd();
    c_23_c_33_vreg.v = _mm_setzero_pd();

    for(int p = 0; p<k; p++){
        a_0p_a_1p_vreg.v = _mm_load_pd((double*) &A(0,p));
        a_2p_a_3p_vreg.v = _mm_load_pd((double*) &A(2,p));

        b_p0_vreg.v = _mm_loaddup_pd((double*) b_p0_ptr++);
        b_p1_vreg.v = _mm_loaddup_pd((double*) b_p1_ptr++);
        b_p2_vreg.v = _mm_loaddup_pd((double*) b_p2_ptr++);
        b_p3_vreg.v = _mm_loaddup_pd((double*) b_p3_ptr++);

        c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
        c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
        c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
        c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

        c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
        c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
        c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
        c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
    }

    C(0,0) += c_00_c_10_vreg.d[0];
    C(0,1) += c_01_c_11_vreg.d[0];
    
    C(0,2) += c_02_c_12_vreg.d[0];
    C(0,3) += c_03_c_13_vreg.d[0];

    C(1,0) += c_00_c_10_vreg.d[1];
    C(1,1) += c_01_c_11_vreg.d[1];
    
    C(1,2) += c_02_c_12_vreg.d[1];
    C(1,3) += c_03_c_13_vreg.d[1];

    C(2,0) += c_20_c_30_vreg.d[0];
    C(2,1) += c_21_c_31_vreg.d[0];
    
    C(2,2) += c_22_c_32_vreg.d[0];
    C(2,3) += c_23_c_33_vreg.d[0];

    C(3,0) += c_20_c_30_vreg.d[1];
    C(3,1) += c_21_c_31_vreg.d[1];
    
    C(3,2) += c_22_c_32_vreg.d[1];
    C(3,3) += c_23_c_33_vreg.d[1];
}

void PackMatrixA(int k, double *a, int lda, double * a_to){
    for(int j=0;j<k;j++){
        double *a_ij_ptr = &A(0,j);
        *a_to++ = *a_ij_ptr;
        *a_to++ = *(a_ij_ptr+1);
        *a_to++ = *(a_ij_ptr+2);
        *a_to++ = *(a_ij_ptr+3);
    }
}

void InnerKernel_(int m, int n, int k, double * a, int lda, double *b, int ldb, double *c, int ldc){
    double packedA[m*k];
    for(int j=0;j<n;j+=4){
        for(int i=0;i<m;i+=4){
            if(j==0) PackMatrixA(k,&A(i,0),lda, &packedA[i*k]);
            AddDot4x4(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
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

    int i,j,p,pb,ib;

	for(p=0;p<k;p+=kc){
        pb = min(k-p,kc);
        for(i=0;i<m;i+=mc){
            ib = min(m-i,mc);
            InnerKernel_(ib,n,pb,&A(i,p),lda, &B(p,0),ldb,&C(i,0), ldc);
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
    
    printf("Optimization 13 : Dot product took %f seconds GFLOPS : %f\n",duration,gflops/duration);
    return 0;
}