#include <gemm.h>

#define A_row(i,j) a[(i)*lda+(j)]
#define B_row(i,j) b[(i)*ldb+(j)]
#define C_row(i,j) c[(i)*ldc+(j)]

#define A_col(i,j) a[(j)*lda+(i)]
#define B_col(i,j) b[(j)*ldb+(i)]
#define C_col(i,j) c[(j)*ldc+(i)]

void serial_init(int m, int n, double * a, int lda, int type){
    int count = 1;
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
			if (type==0) 
            	A_row(i,j) = count++;
			else
				A_col(i,j) = count++;
		}
    }
}

void random_init(int m, int n, double * a, int lda, int type){
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
			if (type==0) 
            	A_row(i,j) = 2.0 * drand48() - 1.0;
			else
				A_col(i,j) = 2.0 * drand48() - 1.0;
		}
    }
}

void display(double * matrix, int m, int n, int type){
	if (type==0){
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				printf("%f ",matrix[i*m+j]);
			}
			printf("\n");
		}
	}
	else{
		for(int j=0;j<n;j++){
			for(int i=0;i<m;i++){
				printf("%f ",matrix[j*m+i]);
			}
			printf("\n");
		}
	}
    return;
}

void matmul_row(int m, int n, int k, double * a, int lda, double * b, int ldb, double * c, int ldc){
	/*
	Computes the matrix multiplication of A and B and stores in C.
	
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

	Matrix Multiply Condition
	-------------------------
		Number of Columns in first matrix must be equal to the number of rows in the second matrix
	*/

	if(a==NULL || b==NULL || c==NULL){
		printf("Argument Error : One of the input arguments to matmul() was NULL\n");
		return;
	}

	for(int i = 0; i<m;i++){
		for(int j = 0; j<n;j++){
			for(int p = 0;p<k;p++){
				C_row(i,j) += A_row(i,p) * B_row(p,j);
			}
		}
	}
	return;
}

void matmul_col(int m, int n, int k, double * a, int lda, double * b, int ldb, double * c, int ldc){
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

	Matrix Multiply Condition
	-------------------------
		Number of Columns in first matrix must be equal to the number of rows in the second matrix
	*/

	if(a==NULL || b==NULL || c==NULL){
		printf("Argument Error : One of the input arguments to matmul() was NULL\n");
		return;
	}

	for(int i = 0; i<m;i++){
		for(int j = 0; j<n;j++){
			for(int p = 0;p<k;p++){
				C_col(i,j) = C_col(i,j) + A_col(i,p) * B_col(p,j);
			}
		}
	}
	return;
}


int main(){

	int m = 2000;
	int n = 2000;
	int k = 2000;

	double * A = (double*)malloc(m*k*sizeof(double)); //A = (m,k)
	double * B = (double*)malloc(k*n*sizeof(double)); //B = (k,n)
	double * C = (double*)malloc(m*n*sizeof(double)); //C = (m,n)
	double * D = (double*)malloc(m*n*sizeof(double)); //D = (m,n)
	
	struct timeval start_row,start_col,finish_row,finish_col;
    double gflops = 2.0 * m*n*k * 1.0e-09;
    srand((unsigned)time(NULL));

    if(A==NULL || B==NULL || C==NULL){
        printf("Out of Memory!\n");
        exit(EXIT_FAILURE);
    }

    random_init(m,k,A,m,0);
    random_init(k,n,B,k,0);

    gettimeofday(&start_row, NULL);
    matmul_row(m,n,k,A,m,B,k,C,m);
    gettimeofday(&finish_row, NULL);

    double duration_row = ((double)(finish_row.tv_sec-start_row.tv_sec)*1000000 + (double)(finish_row.tv_usec-start_row.tv_usec)) / 1000000;
	
	random_init(m,k,A,m,1);
	random_init(k,n,B,k,1);

	gettimeofday(&start_col, NULL);
    matmul_col(m,n,k,A,m,B,k,D,m);
    gettimeofday(&finish_col, NULL);

	double duration_col = ((double)(finish_col.tv_sec-start_col.tv_sec)*1000000 + (double)(finish_col.tv_usec-start_col.tv_usec)) / 1000000;
	
	printf("Naive : Dot product with row major order took %f seconds GFLOPS : %f\n",duration_row,gflops/duration_row);
	printf("Naive : Dot product with col major order took %f seconds GFLOPS : %f\n",duration_col,gflops/duration_col);
	
	free(A);
	free(B);
	free(C);
	free(D);
	return 0;
}