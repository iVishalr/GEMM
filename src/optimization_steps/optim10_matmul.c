#include <gemm.h>

#define A(i,j) a[(j)*lda+(i)]
#define B(i,j) b[(j)*ldb+(i)]
#define C(i,j) c[(j)*ldc+(i)]

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

    register double c_00, c_01, c_02, c_03, c_10, c_11, c_12, c_13, c_20, c_21, c_22, c_23, c_30, c_31, c_32, c_33, a_0p, a_1p, a_2p, a_3p, b_0p_reg, b_1p_reg, b_2p_reg, b_3p_reg;
    double * b_0p_ptr, *b_1p_ptr, *b_2p_ptr, *b_3p_ptr;

    c_00 = 0.0;   
    c_01 = 0.0;   
    c_02 = 0.0;   
    c_03 = 0.0;
    c_10 = 0.0;   
    c_11 = 0.0;   
    c_12 = 0.0;   
    c_13 = 0.0;
    c_20 = 0.0;   
    c_21 = 0.0;   
    c_22 = 0.0;   
    c_23 = 0.0;
    c_30 = 0.0;   
    c_31 = 0.0;   
    c_32 = 0.0;   
    c_33 = 0.0;

    b_0p_ptr = &B(0,0);
    b_1p_ptr = &B(0,1);
    b_2p_ptr = &B(0,2);
    b_3p_ptr = &B(0,3);

    // #pragma omp parallel for num_threads(4)
    for(int p=0;p<k;p++){
        a_0p = A(0,p);
        a_1p = A(1,p);
        a_2p = A(2,p);
        a_3p = A(3,p);

        b_0p_reg = *b_0p_ptr++;
        b_1p_reg = *b_1p_ptr++;
        b_2p_reg = *b_2p_ptr++;
        b_3p_reg = *b_3p_ptr++;

        c_00 += a_0p * b_0p_reg;
        c_10 += a_1p * b_0p_reg;

        c_01 += a_0p * b_1p_reg;
        c_11 += a_1p * b_1p_reg;
        
        c_02 += a_0p * b_2p_reg;
        c_12 += a_1p * b_2p_reg;
        
        c_03 += a_0p * b_3p_reg;
        c_13 += a_1p * b_3p_reg;


        c_20 += a_2p * b_0p_reg;
        c_30 += a_3p * b_0p_reg;
        
        c_21 += a_2p * b_1p_reg;
        c_31 += a_3p * b_1p_reg;
        
        c_22 += a_2p * b_2p_reg;
        c_32 += a_3p * b_2p_reg;
        
        c_23 += a_2p * b_3p_reg;
        c_33 += a_3p * b_3p_reg;   
    }

    C(0,0) += c_00;
    C(0,1) += c_01;
    C(0,2) += c_02;
    C(0,3) += c_03;

    C(1,0) += c_10;
    C(1,1) += c_11;
    C(1,2) += c_12;
    C(1,3) += c_13;

    C(2,0) += c_20;
    C(2,1) += c_21;
    C(2,2) += c_22;
    C(2,3) += c_23;

    C(3,0) += c_30;
    C(3,1) += c_31;
    C(3,2) += c_32;
    C(3,3) += c_33;
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
    // #pragma omp parallel for num_threads(4)
	for(int j=0; j<n; j+=4){ //Loop over the columns of C with stride 4
        for(int i=0;i<m;i+=4){ //Loop over the rows of C
            AddDot4x4(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
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
    
    printf("Optimization 10 : Dot product took %f seconds GFLOPS : %f\n",duration,gflops/duration);
    return 0;
}