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

void AddDot1x4(int k, double * a, int lda, double * b, int ldb, double * c, int ldc){
    int p;

    register double c_00, c_01, c_02, c_03, a_0p;
    double *b_00_ptr, *b_01_ptr, *b_02_ptr, *b_03_ptr;

    b_00_ptr = &B(0,0);
    b_01_ptr = &B(0,1);
    b_02_ptr = &B(0,2);
    b_03_ptr = &B(0,3);

    c_00 = 0.0f;
    c_01 = 0.0f; 
    c_02 = 0.0f; 
    c_03 = 0.0f;

    for(p=0;p<k;p+=4){
        a_0p = A(0,p);

        c_00 += a_0p * *b_00_ptr;
        c_01 += a_0p * *b_01_ptr;
        c_02 += a_0p * *b_02_ptr;
        c_03 += a_0p * *b_03_ptr;

        a_0p = A(0,p+1);

        c_00 += a_0p * *(b_00_ptr+1);
        c_01 += a_0p * *(b_01_ptr+1);
        c_02 += a_0p * *(b_02_ptr+1);
        c_03 += a_0p * *(b_03_ptr+1);

        a_0p = A(0,p+2);

        c_00 += a_0p * *(b_00_ptr+2);
        c_01 += a_0p * *(b_01_ptr+2);
        c_02 += a_0p * *(b_02_ptr+2);
        c_03 += a_0p * *(b_03_ptr+2);

        a_0p = A(0,p+3);

        c_00 += a_0p * *(b_00_ptr+3);
        c_01 += a_0p * *(b_01_ptr+3);
        c_02 += a_0p * *(b_02_ptr+3);
        c_03 += a_0p * *(b_03_ptr+3);

        b_00_ptr += 4;
        b_01_ptr += 4;
        b_02_ptr += 4;
        b_03_ptr += 4;
    }

    C(0,0) += c_00;
    C(0,1) += c_01;
    C(0,2) += c_02;
    C(0,3) += c_03;
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

	for(int j=0; j<n; j+=4){ //Loop over the columns of C with stride 4
        for(int i=0;i<m;i++){ //Loop over the rows of C
            AddDot1x4(k,&A(i,0),lda,&B(0,j),ldb,&C(i,j),ldc);
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
    
    printf("Optimization 5 : Dot product took %f seconds GFLOPS : %f\n",duration,gflops/duration);
    return 0;
}