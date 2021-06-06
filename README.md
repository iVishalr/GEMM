# General Matrix Multiplication (GEMM)

<p align="center">
  <img src="https://petewarden.files.wordpress.com/2015/04/gemm_corrected.png" alt="matrix-multiplication-image" height="250px" width="600px"></img>
</p>

This implementation provides fast matrix multiplication for multiplying two square matrices. This method is known as the general matrix multiplication (GEMM). Many scientific computing libraries like Numpy, BLAS, etc., use a slight modified version of this algorithm. This implementation works only on square matrices. This is done to avoid making the algorithm too complicated to handle rectangular matrices. 

This implementation deals with multiplying two sqaured matrices ordered in column major order. In Linear Algebra you order everything according to columns. Hence we will also order it column wise. The two square matrices must have dimensions that is divisible by 4. This is because we are using a 4x4 kernel to compute the dot products of sub matrices. 

This implementation achieves good performance using only a single thread. Majority of the performance comes from storing the 4x4 kernel in vector registers in the CPU and perform the dot products. Vector registers use instructions provided by `SSE3`(Streaming SIMD Extension 3). We slice and dice the original large matrices in such a way that the slices fit in L1 and L2 caches which means that CPU doesn't need to wait for data to be loaded to cache before performing the dot products. This gives us a decent performance boost when we compare the naive approach to compute the matrix multiplication.

## Features

1. Fast Matrix Multiplication of Square Matrices
2. Register, Cache blocking
3. No dependencies

## Building the Project

Clone this repository by typing the following commands in terminal,

```bash
$ git clone https://github.com/iVishalr/GEMM.git
$ cd GEMM
```

Next, we need to compile the code. Typing the following command will compile the code.

```bash
$ make
```

To execute GEMM, type the following in terminal,

```bash
$ ./gemm
```

Additionally, there are other intermediate optimization files included. Compiling and executing them shows the performance improvements with slight modifications to the implementation of the algorithm. To run this kind of a benchmark, type the following command in terminal,

```bash
$ make optim
``` 

This will run all the files present in `src/optimization_steps/` folder and you can see the difference in performance as it exxecutes.

To clean the object files, type the following command in terminal,

```bash
$ make clean
```

## Limitations

1. Works only on Square Matrices whose dimensions are divisible by 4. GEMM uses a 4x4 kernel to compute the dot product of submatrices. Official implementations of GEMM use multiple kernels optimized to different CPU architectures. Too complicated to implement :P
2. This is a single threaded implementation of GEMM. Getting multi-threaded implementation of GEMM requires coding in much lower level and different kernel specific optimizations. Numpy, BLAS will be much faster than what this implementation provides.

## License

MIT
