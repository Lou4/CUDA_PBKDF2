#include <stdio.h>
#include <assert.h>
#include "utils.h"


/**
 * Given the total size of the matrix of thread you and the total size of the matrix you need to handle
 * are going to spawn, the function print a grafical rappresentation to show which thread will remain idle.
 *
 * Params:
 * MATRIX_ELEMS_IN_A_ROW: number of thread in a block's row * number of blocks
 * MATRIX_ELEMS_IN_A_COL: number of thread in a block's column * number of blocks
 * DATA_ROW_ELEMS: total element in a row in your data matrix
 * DATA_COL_ELEMS: total element in a col in your data matrix
 * THREAD_PER_BLOCK_X: number of thread in a block's row
 * THREAD_PER_BLOCK_Y: number of thread in a block's column
 */
void print_thread_logic(int const MATRIX_X, int const MATRIX_Y, int const Nx, int const Ny, int const Bx, int const By){
	int x, y;
	int idx;
	int thread_active = 0, thread_idle = 0;
	for(y = 0; y < MATRIX_Y; y++){
		//If it's going to be print the first element of a new block [COL]
		if(x != 0 && y % By == 0)
			printf("\n");

		for(x = 0; x < MATRIX_X; x++){
			//If it's going to be print the first element of a new block [ROW]
			if(x != 0 && x % Bx == 0)
				printf("%4s", "");

			//c > number of column of the data matrix
			if(y >= Ny){
				printf("%4d", y);
				thread_idle++;
			}

			//r > number of row of the data matrix
			else if(x >= Nx){
				printf("%4d", y);
				thread_idle++;
			}

			//Non idle thread
			else{
				idx = (y*Nx)+x; //For future use
				printf("%4s", "--");
				thread_active++;
			}
		}
		printf("\n");
	}

	printf("Thread Matrix %dx%d, Data Matrix %dx%d\n", MATRIX_Y, MATRIX_X, Ny, Nx);
	printf("Thread Active: %d, Thread Idle: %d\n", thread_active, thread_idle);
}

/**
 * TODO: doc
 */
void calculate_grid_block_dim(int *gx, int *gy, int *bx, int *by, int const R, int const C){
	cudaDeviceProp cudaProp;
	CHECK(cudaGetDeviceProperties(&cudaProp, 0));

	int maxBlock_x = cudaProp.maxThreadsDim[0];
	int maxBlock_y = cudaProp.maxThreadsDim[1];
	int maxGrid_x = cudaProp.maxGridSize[0];
	int maxGrid_y = cudaProp.maxGridSize[1];

	if(R*C <= maxBlock_x){
		*gx = 1;
		*gy = 1;
		*bx = R;
		*by = C;
	}else{
		*gx = (R / BLOCK_DIM_X ) + 1;
		*gy = (C / BLOCK_DIM_Y ) + 1;
		*bx = BLOCK_DIM_X;
		*by = BLOCK_DIM_Y;
	}

	if(*gx > maxGrid_x || *gy > maxGrid_y){
		printf("Error, matrix too big.\n");
		exit(EXIT_FAILURE);
	}
	printf("grid_x: %d, grid_y: %d, block_x: %d, block_y: %d\n", *gx, *gy, *bx, *by);
	return;
}

void print_linear_matrix(int *m, int const R, int const C){
	printf("- - - PRINT MATRIX %dx%d - - -\n", R, C);
	for(int r=0; r<R; r++){
		for(int c=0; c<C; c++){
			printf("%5d  ", m[r*C+c]);
		}
		printf("\n");
	}
	printf("- - - - - - - - - - - - - - - -\n\n");
}

void print_linear_matrix(float *m, int const R, int const C){
	printf("- - - PRINT MATRIX %dx%d - - -\n", R, C);
	for(int r=0; r<R; r++){
		for(int c=0; c<C; c++){
			printf("%5.5f  ", m[r*C+c]);
		}
		printf("\n");
	}
	printf("- - - - - - - - - - - - - - - -\n\n");
}



/*./Lesson03_Profiling 80 10 10 40
 * Thread Matrix 96x64, Data Matrix 80x40
 * grid_x: 9, grid_y: 11, block_x: 12, block_y: 10
 *
 *
 * Thread Matrix 110x108, Data Matrix 100x100
Thread Active: 10000, Thread Idle: 1880grid_x: 9, grid_y: 11, block_x: 12, block_y: 10
[   0][   1][   2][   3][   4][   5][   6][   7][   8][   9][  10][  11] .. [  12][  13][  14][  15][  16][  17][  18][  19][  20][  21][  22][  23] .. [  24][  25][  26][  27][  28][  29][  30][  31][  32][  33][  34][  35] .. [  36][  37][  38][  39][  40][  41][  42][  43][  44][  45][  46][  47] .. [  48][  49][  50][  51][  52][  53][  54][  55][  56][  57][  58][  59] .. [  60][  61][  62][  63][  64][  65][  66][  67][  68][  69][  70][  71] .. [  72][  73][  74][  75][  76][  77][  78][  79][  80][  81][  82][  83] .. [  84][  85][  86][  87][  88][  89][  90][  91][  92][  93][  94][  95] .. [  96][  97][  98][  99][100][101][102][103][104][105][106][107]
[   0][   1][   2][   3][   4][   5][   6][   7][   8][   9][  10][  11] .. [  12][  13][  14][  15][  16][  17][  18][  19][  20][  21][  22][  23] .. [  24][  25][  26][  27][  28][  29][  30][  31][  32][  33][  34][  35] .. [  36][  37][  38][  39][  40][  41][  42][  43][  44][  45][  46][  47] .. [  48][  49][  50][  51][  52][  53][  54][  55][  56][  57][  58][  59] .. [  60][  61][  62][  63][  64][  65][  66][  67][  68][  69][  70][  71] .. [  72][  73][  74][  75][  76][  77][  78][  79][  80][  81][  82][  83] .. [  84][  85][  86][  87][  88][  89][  90][  91][  92][  93][  94][  95] .. [  96][  97][  98][  99][100][101][102][103][104][105][106][107]
[   0][   1][   2][   3][   4][   5][   6][   7][   8][   9][  10][  11] .. [  12][  13][  14][  15][  16][  17][  18][  19][  20][  21][  22][  23] .. [  24][  25][  26][  27][  28][  29][  30][  31][  32][  33][  34][  35] .. [  36][  37][  38][  39][  40][  41][  42][  43][  44][  45][  46][  47] .. [  48][  49][  50][  51][  52][  53][  54][  55][  56][  57][  58][  59] .. [  60][  61][  62][  63][  64][  65][  66][  67][  68][  69][  70][  71] .. [  72][  73][  74][  75][  76][  77][  78][  79][  80][  81][  82][  83] .. [  84][  85][  86][  87][  88][  89][  90][  91][  92][  93][  94][  95] .. [  96][  97][  98][  99][100][101][102][103][104][105][106][107]
- - - PRINT MATRIX 100x100 - - -*/



int checkDeviceProperties(int b_x, int b_y, int b_z, int g_x, int g_y, int g_z, int globalMem, int dev){

	cudaDeviceProp cudaDeviceProp;
	cudaGetDeviceProperties(&cudaDeviceProp, 0);

	int const MAX_BLOCK_X = cudaDeviceProp.maxThreadsDim[0];
	int const MAX_BLOCK_Y = cudaDeviceProp.maxThreadsDim[1];
	int const MAX_BLOCK_Z = cudaDeviceProp.maxThreadsDim[2];
	int const MAX_GRID_X = cudaDeviceProp.maxGridSize[0];
	int const MAX_GRID_Y = cudaDeviceProp.maxGridSize[1];
	int const MAX_GRID_Z = cudaDeviceProp.maxGridSize[2];
	int const MAX_THREAD_PER_BLOCK = cudaDeviceProp.maxThreadsPerBlock;
	int const MAX_GLOBAL_MEMORY_AVILABLE = cudaDeviceProp.totalGlobalMem;

	printf("\n- - - - - - - checkDeviceProperties() - - - - - - \n");
	printf("block_x: %d\n", b_x);
	printf("block_y: %d\n", b_y);
	printf("block_z: %d\n", b_z);
	printf("grid_x: %d\n", g_x);
	printf("grid_y: %d\n", g_y);
	printf("grid_z: %d\n", g_z);
	printf("global memory usage: %d Bytes\n", globalMem);
	printf("- - - - - - - - - - - - - - - - - - - - - - - - - \n\n");

	assert(MAX_BLOCK_X > b_x);
	assert(MAX_BLOCK_Y > b_y);
	assert(MAX_BLOCK_Z > b_z);
	assert(MAX_GRID_X > g_x);
	assert(MAX_GRID_Y > g_y);
	assert(MAX_GRID_Z > g_z);
	assert(MAX_THREAD_PER_BLOCK > b_x*b_y*b_z);
	assert(MAX_GLOBAL_MEMORY_AVILABLE > globalMem);
}

int* createInitializedArray(int const LEN, int const RANGE, int const OFFSET){
	int* a = (int*) calloc(LEN, sizeof(int));
	for(int i = 0; i < LEN; i++){
		if(OFFSET < 0)
			a[i] = i+1;
		else
			a[i] = (rand() % RANGE) + OFFSET;
	}

	return a;
}

float* createInitializedArray(int const LEN, int const RANGE, float const OFFSET){
	float* a = (float*) calloc(LEN, sizeof(float));
	for(int i = 0; i < LEN; i++){
		if(OFFSET < 0)
			a[i] = i+1;
		else
			a[i] = (rand() % RANGE) + OFFSET;
	}

	return a;
}

int isPowOfTwo(int num){
	return (num != 0) && ((num & (num-1)) == 0);
}

char* prettyPrintNumber(long n){
	char tmp[20];
	char *number = (char*) calloc(20, sizeof(char));

	int r;
	int i = 0;
	int j = 0;

	do{
		r = n % 10;
		n /= 10;

		tmp[i++] = r + 48;
		j++;

		if(j % 3 == 0 && n != 0)
			tmp[i++] = '.';

	}while(n > 0);

	i--;

	int el = 0;
	for(; i>=0; i--)
		number[el++] = tmp[i];
	number[el] = 0;

	return number;
}

int checkArchitecturalBoundaries(int const DEV, int const Gx, int const Gy, int const Bx, int const By, int const N_BYTES_GLOBAL_MEMORY_USAGE, int SMEM_REQUIRED, int PRINT){

<<<<<<< HEAD
	if(PRINT) printf("\n- - - START CHECKING - - -\n");
=======
	int const NO_VERBOSE = !PRINT;
	printf("\n- - - START CHECKING - - -\n");
>>>>>>> bfbfddfc730f8b0abf1d8d07500a0d9da9264712

	cudaDeviceProp cudaDeviceProp;
	CHECK(cudaGetDeviceProperties(&cudaDeviceProp, DEV));

	if(PRINT) printf("Max thread per block: %d  -  required %d\n", cudaDeviceProp.maxThreadsPerBlock, By*Bx);
	assert(cudaDeviceProp.maxThreadsPerBlock > Bx * By);
<<<<<<< HEAD
=======
	if(NO_VERBOSE) printf("max thread per block . . . OK\n");
>>>>>>> bfbfddfc730f8b0abf1d8d07500a0d9da9264712

	if(PRINT) printf("Max block dim (x: %d, y: %d)  -  required (%d, %d)\n", cudaDeviceProp.maxThreadsDim[0], cudaDeviceProp.maxThreadsDim[1], Bx, By);
	assert(cudaDeviceProp.maxThreadsDim[0] > Bx);
	assert(cudaDeviceProp.maxThreadsDim[1] > By);
<<<<<<< HEAD
=======
	if(NO_VERBOSE) printf("block dimensions . . . OK\n");
>>>>>>> bfbfddfc730f8b0abf1d8d07500a0d9da9264712

	if(PRINT) printf("Max grid dim (x: %d, y: %d)  -  required (%d, %d)\n", cudaDeviceProp.maxGridSize[0], cudaDeviceProp.maxGridSize[1], Gx, Gy);
	assert(cudaDeviceProp.maxGridSize[0] > Gx);
	assert(cudaDeviceProp.maxGridSize[1] > Gy);
<<<<<<< HEAD

	if(PRINT) printf("Global memory available: %d MegaByte  -  required %d MegaByte\n", cudaDeviceProp.totalGlobalMem/1024/1024, N_BYTES_GLOBAL_MEMORY_USAGE/1024/1024);
	assert(cudaDeviceProp.totalGlobalMem > N_BYTES_GLOBAL_MEMORY_USAGE);

	if(PRINT) printf("Shared memory available: %d KiloByte - required %d KiloByte\n", cudaDeviceProp.sharedMemPerBlock/1024, SMEM_REQUIRED/1024);
	assert(cudaDeviceProp.sharedMemPerBlock > SMEM_REQUIRED);

	if(PRINT) printf("- - - END CHECKING - - -\n\n");
=======
	if(NO_VERBOSE) printf("grid dimension . . . OK\n");

	if(PRINT) printf("Global memory available: %d MegaByte  -  required %d MegaByte\n", cudaDeviceProp.totalGlobalMem/1024/1024, N_BYTES_GLOBAL_MEMORY_USAGE/1024/1024);
	assert(cudaDeviceProp.totalGlobalMem > N_BYTES_GLOBAL_MEMORY_USAGE);
	if(NO_VERBOSE) printf("global memory usage . . . OK\n");

	if(PRINT) printf("Shared memory available: %d KiloByte - required %d KiloByte\n", cudaDeviceProp.sharedMemPerBlock/1024, SMEM_REQUIRED/1024);
	assert(cudaDeviceProp.sharedMemPerBlock > SMEM_REQUIRED);
	if(NO_VERBOSE) printf("smem usage . . . OK\n");

	printf("- - - END CHECKING - - -\n\n");
>>>>>>> bfbfddfc730f8b0abf1d8d07500a0d9da9264712
}
