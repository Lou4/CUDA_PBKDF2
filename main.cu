#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "my_C_lib/utils.h"
#include "my_C_lib/CPU_time.h"
#include "hashlib/hmac-sha1.cuh"
#include "hashlib/sha1.cuh"

#define H_LEN 20 // Length in Bytes of the PRF functions' output
#define DEV 0
#define intDivCeil(n, d) ((n + d - 1) / d)

int DEBUG;
char salt[] = "salt";

__constant__ int D_C;
__constant__ int D_SK_LEN;
__constant__ int D_N;


__device__ void actualFunction(char* sk, char* output, int const KERNEL_ID){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;


	if(idx >= D_N)
		return;

	globalChars globalChars;
	uint8_t salt[H_LEN] = "salt";
	int saltLen = 4 + (2 * sizeof(int));
	int *ptr = (int*)&salt[4];
	uint8_t acc[H_LEN];
	uint8_t buffer[H_LEN];


	cudaMemcpyDevice(&ptr[0], &idx, sizeof(int));
	cudaMemcpyDevice(&ptr[1], &KERNEL_ID, sizeof(int));

	/* DEBUG
	if(idx == 1){
		for(int i = 0; i <H_LEN; i++){
			printf("%02x ", salt[i]);
		}
		printf("\n --- \n");
	}*/


	hmac_sha1(sk, D_SK_LEN, salt, saltLen, buffer, &globalChars);
	cudaMemcpyDevice(salt, buffer, H_LEN);
	cudaMemcpyDevice(acc, buffer, H_LEN);
	for(int i = 0; i < D_C; i++){
		hmac_sha1(sk, D_SK_LEN, salt, H_LEN, buffer, &globalChars);
		cudaMemcpyDevice(salt, buffer, H_LEN);

		for(int i = 0; i < H_LEN; i++){
			acc[i] ^= buffer[i];
		}
	}

	/* DEBUG
	if(idx == 0 && KERNEL_ID == 4){
		for(int i = 0; i <H_LEN; i++){
			printf("%02x  ", acc[i]);
		}
		printf("\n --- \n");
	}
	*/

	int index;
	for(int i = 0; i < H_LEN; i++){
		index = idx * H_LEN + i;
		output[index] = acc[i];
	}

}

__global__ void pbkdf2(char* sk, char* output, int *kernelId){
	actualFunction(sk, output, *kernelId);
}


__global__ void pbkdf2_2(char* sk, char* output, int *kernelId){
	actualFunction(sk, output, *kernelId);
}

__global__ void pbkdf2_3(char* sk, char* output){
	actualFunction(sk, output, 0);
}

__global__ void pbkdf2_4(char* sk, char* output, int *kernelId){
	actualFunction(sk, output, *kernelId);
}

__host__ void execution1(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out);
__host__ void execution2(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out);
__host__ void execution3(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out);
__host__ void execution4(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out, int const nStream, int const INDEX);
__host__ void executionSequential(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, struct Data *out);
__host__ void copyValueFromGlobalMemoryToCPUMemory(uint8_t *keys, uint8_t *output, int const NUM, int const LEN, int const OFFSET);
__host__ void printAllKeys(uint8_t *keys, int const LEN, int const NUM);
__host__ void printHeader(int const DK_NUM, int const DK_LEN, int const  BX);
__host__ void printKernelDebugInfo(int const K_ID, int const THREAD_X_K, int const K_BYTES_GENERATED, int const DK_LEN);

/**
 * DF = PBKDF2(PRF,Password, Salt, c, dkLen)
 *
 * DF = T1 || T2 || ... || Tdklen/hlen
 *
 * Ti = U1 xor U2 xor … xor Uc
 *
 * U1 = PRF(Password, Salt || i);
 * U2 = PRF(Password, U1);
 * U3 = PRF(Password, U2);
 * . . .
 * Uc = PRF(Password, Uc-1);
 *
 * One thread will calculata one Ti.
 */

struct Data{
	uint8_t *keys;
	double elapsedKernel;
	double elapsedGlobal;
};


int main(int c, char **v){
	system("clear");
	printf("\t\t\t\t----------- Authors -------------\n");
	printf("\t\t\t\t| Luca Tagliabue, Marco Predari |\n");
	printf("\t\t\t\t---------------------------------\n\n");

	if(c != 7){
		printf("Error !!\n");
		printf("./Project_GPU <Bx> <source_key> <iterations> <len_derived_keys> <num_derived_keys> <DEBUG>\n");

		exit(EXIT_FAILURE);
	}

	//Host var
	int const BX = atoi(v[1]);				// Thread per block
	char const *SOURCE_KEY = v[2];			// Password
	int const SK_LEN = strlen(SOURCE_KEY);	// Password len
	int const C = atoi(v[3]);				// Number of iteration
	int const DK_LEN = atoi(v[4]);			// Derived Keys' length
	int const DK_NUM = atoi(v[5]);			// Number of derived keys we'll generate
	DEBUG = atoi(v[6]);

	int foo;

	assert(isPowOfTwo(DK_LEN) == 1);
	assert(isPowOfTwo(DK_NUM) == 1);

	// One kernel generate one dk, one thread generate one Ti
	int threadsPerKernel = intDivCeil(DK_LEN, H_LEN);		// Threads needed
	int Gx = intDivCeil(threadsPerKernel, BX);				// Calculate Gx

	//Output var
	struct Data *out1, *out2, *out3, *out4, *outS;
	out1 = (struct Data*)malloc(sizeof(struct Data));
	out2 = (struct Data*)malloc(sizeof(struct Data));
	out3 = (struct Data*)malloc(sizeof(struct Data));

	out1->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));
	out2->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));
	out3->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));

	int const N_STREAM[] = {2, 4, 8, 16};
	int const S_LEN = 4;
	out4 = (struct Data*)malloc(sizeof(struct Data) * S_LEN);
	for(int i = 0; i < S_LEN; i++){
		out4[i].keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));
	}

	outS = (struct Data*)malloc(sizeof(struct Data));
	outS->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));

	if(DEBUG){
		printf("SOURCE_KEY: %s\n", SOURCE_KEY);
		printf("C: %d\n", C);
		printf("DK_LEN: %d\n", DK_LEN);
		printf("DK_NUM: %d\n", DK_NUM);
		printf("H_LEN: %d\n", H_LEN);
	}

	printHeader(DK_NUM, DK_LEN, BX);

	CHECK(cudaSetDevice(DEV));

	//Tranfer to CONSTANT MEMORY
	CHECK(cudaMemcpyToSymbol(D_N, &threadsPerKernel, sizeof(int)));	// Thread per kernel
	CHECK(cudaMemcpyToSymbol(D_SK_LEN, &SK_LEN, sizeof(int)));		// Source key len
	CHECK(cudaMemcpyToSymbol(D_C, &C, sizeof(int)));					// Iteration

	// Without Stream
	printf("- - - - - - Execution one, more kernel no stream - - - - - -\n");
	printf("\nTotal number of threads required per kernel: %d\n\n", threadsPerKernel);
	double start = seconds();
	execution1(SOURCE_KEY, C, DK_LEN, DK_NUM, Gx, BX, threadsPerKernel, out1);
	out1->elapsedGlobal = seconds() - start;
	printf("- - - - - - - End execution one - - - - - - - - - - - - - -\n");

	printf("\n\n\n* * * **************************************************************************************** * * *\n\n\n");

	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX);

	//Tranfer to CONSTANT MEMORY
	CHECK(cudaMemcpyToSymbol(D_N, &threadsPerKernel, sizeof(int)));	// Thread per kernel
	CHECK(cudaMemcpyToSymbol(D_SK_LEN, &SK_LEN, sizeof(int)));		// Source key len
	CHECK(cudaMemcpyToSymbol(D_C, &C, sizeof(int)));					// Iteration

	// With Stream
	printf("- - - - - - Execution two, with stream - - - - - -\n");
	printf("\nTotal number of threads required per kernel: %d\n\n", threadsPerKernel);
	start = seconds();
	execution2(SOURCE_KEY, C, DK_LEN, DK_NUM, Gx, BX, threadsPerKernel, out2);
	out2->elapsedGlobal = seconds() - start;
	printf("- - - - - - - - End execution two - - - - - - - - \n");


	printf("\n\n\n* * * **************************************************************************************** * * *\n\n\n");

	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX);

	// One kernel generate ALL dk, one thread generate one Ti
	threadsPerKernel = intDivCeil((DK_LEN * DK_NUM), H_LEN);	// Threads needed
	Gx = intDivCeil(threadsPerKernel, BX);				// Calculate Gx

	//Tranfer to CONSTANT MEMORY
	CHECK(cudaMemcpyToSymbol(D_N, &threadsPerKernel, sizeof(int)));	// Thread per kernel
	CHECK(cudaMemcpyToSymbol(D_SK_LEN, &SK_LEN, sizeof(int)));		// Source key len
	CHECK(cudaMemcpyToSymbol(D_C, &C, sizeof(int)));					// Iteration

	printf("- - - - - - Execution three, one kernel no stream - - - - - -\n");
	printf("\nTotal number of threads required per kernel: %d\n\n", threadsPerKernel);
	start = seconds();
	execution3(SOURCE_KEY, C, DK_LEN, DK_NUM, Gx, BX, threadsPerKernel, out3);
	out3->elapsedGlobal = seconds() - start;
	printf("- - - - - - - End execution three - - - - - - - - - - - - - -\n");


	printf("\n\n\n* * * **************************************************************************************** * * *\n\n\n");
	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX);

	printf("- - - - - - Execution four, rational use of stream - - - - - -\n");
	for(int i = 0; i < S_LEN; i++){
		threadsPerKernel = intDivCeil((DK_LEN * DK_NUM), (H_LEN*N_STREAM[i]));	// Threads needed
		Gx = intDivCeil(threadsPerKernel, BX);									// Calculate Gx
		printf("Total Bytes: %d\n", DK_LEN * DK_NUM);
		printf("N_STREAM: %d\n", N_STREAM[i]);
		printf("\nTotal number of threads required per kernel: %d\n", threadsPerKernel);
		printf("Every stream generate %d Bytes.\n\n", threadsPerKernel * H_LEN);

		//Tranfer to CONSTANT MEMORY
		CHECK(cudaMemcpyToSymbol(D_N, &threadsPerKernel, sizeof(int)));	// Thread per kernel
		CHECK(cudaMemcpyToSymbol(D_SK_LEN, &SK_LEN, sizeof(int)));		// Source key len
		CHECK(cudaMemcpyToSymbol(D_C, &C, sizeof(int)));					// Iteration

		start = seconds();
		execution4(SOURCE_KEY, C, DK_LEN, DK_NUM, Gx, BX, threadsPerKernel, out4, N_STREAM[i], i);
		out4[i].elapsedGlobal = seconds() - start;
		printf("\n\n\n\n");
	}
	printf("- - - - - - - End execution four - - - - - - - - - - - - - -\n");

	printf("\n\n\n* * * **************************************************************************************** * * *\n\n\n");
	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX);

	printf("- - - - - - Last but not least execution, SEQUENTIAL - - - - - -\n");
	start = seconds();
	executionSequential(SOURCE_KEY, C, DK_LEN, DK_NUM, outS);
	outS->elapsedGlobal = seconds() - start;
	printf("- - - - - - - - End last execution - - - - - - - - - - - - - - -\n");

	printf("\n\n\n* * * **************************************************************************************** * * *\n\n\n");
	printf("Press enter to continue . . .");
	//scanf("%d", &foo);


	// Check correctness first two execution
	printf("check correctness . . .\n");
	for(int i=0; i<DK_NUM; i++){
		if (DEBUG) printf("check %d key (len %d Bytes)...", i, sizeof(uint8_t)*DK_LEN);
		assert(memcmp(out1->keys, out2->keys, DK_NUM * DK_LEN * sizeof(uint8_t)) == 0);
		if (DEBUG) printf("ok\n");
	}

	printf("\n\n\n- - - - - - - - - - RESULT - - - - - - - - - - - - - - \n");
	printf("  Witout stream takes %f millisec\n", out1->elapsedGlobal);
	printf("  With stream takes %f millisec\n", out2->elapsedGlobal);
	printf("  One kernel takes %f millisec\n", out3->elapsedGlobal);
	for(int i = 0; i < S_LEN; i++){
		printf("  Stream with rational takes %f millisec\n", out4[i].elapsedGlobal);
	}
	printf("  Sequential takes %f millisec\n", outS->elapsedGlobal);
	printf("\n");

	printf("  Witout stream kernel and transfert  takes %f millisec\n", out1->elapsedKernel);
	printf("  With stream kernel and transfert takes %f millisec\n", out2->elapsedKernel);
	printf("  One kernel and transfert takes %f millisec\n", out3->elapsedKernel);
	for(int i = 0; i < S_LEN; i++){
		printf("  Stream with kernel and transfert takes %f millisec\n", out4[i].elapsedKernel);
	}
	printf("  Sequential haven't kernel so ... \n");

	printf("\n");

	printf("  With stream gain: %c %2lf\n", 37, 100-((100*out2->elapsedGlobal)/out1->elapsedGlobal));
	printf("  With stream gain kernel and transfert: %c %2lf\n", 37, 100-((100*out2->elapsedKernel)/out1->elapsedKernel));
	printf("  One kernel gain: %c %2lf\n", 37, 100-((100*out3->elapsedGlobal)/out2->elapsedGlobal));
	printf("  One kernel gain kernel and transfert: %c %2lf\n", 37, 100-((100*out3->elapsedKernel)/out2->elapsedKernel));
	printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");
	return 0;
}



__host__ void execution1(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out){

	//Alloc and init CPU memory
	int const N_BYTES_OUTPUT =  THREAD_X_KERNEL * H_LEN * DK_NUM * sizeof(char);
	char	 *output = (char*)malloc(N_BYTES_OUTPUT);
	memset(output, 0, N_BYTES_OUTPUT);

	printf("N_BYTES_OUTPUT: %s Bytes\n", prettyPrintNumber(N_BYTES_OUTPUT));

	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT, 0, 0);

	//Device var
	char *d_output;
	char *d_sk;
	int *d_kernelId;


	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);


	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY
	CHECK(cudaMalloc((void**)&d_kernelId, sizeof(int)));

	// Output var
	CHECK(cudaMalloc((void**)&d_output, N_BYTES_OUTPUT));
	CHECK(cudaMemset(d_output, 0, N_BYTES_OUTPUT));

	// Source Key
	int N_BYTES_SK = (strlen(SOURCE_KEY) + 1) * sizeof(char); // +1 because of null end char
	CHECK(cudaMalloc((void**)&d_sk, N_BYTES_SK));
	CHECK(cudaMemcpy(d_sk, SOURCE_KEY, N_BYTES_SK, cudaMemcpyHostToDevice));


	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

	//Starting kernel
	printf("Starting %d kernels with %s threads each (%d threads needed)...\n", DK_NUM, prettyPrintNumber(block.x*grid.x), THREAD_X_KERNEL);
	int index;
	double start = seconds();
	for(int i = 0; i < DK_NUM; i++){
		index = i* THREAD_X_KERNEL * H_LEN;
		printKernelDebugInfo(i, THREAD_X_KERNEL, THREAD_X_KERNEL*H_LEN, DK_LEN);

		CHECK(cudaMemcpy(d_kernelId, &i, sizeof(int), cudaMemcpyHostToDevice));
		pbkdf2<<<grid, block>>>(d_sk, &d_output[index], d_kernelId);
		CHECK(cudaMemcpy(&output[index], &d_output[index], DK_LEN * sizeof(char), cudaMemcpyDeviceToHost));

		printf("Copy %d° key, %d Bytes starting index output[%d]\n\n", i+1, DK_LEN*sizeof(char), index);
	}

	CHECK(cudaDeviceSynchronize());

	out->elapsedKernel = seconds() - start;
	printf("%d kernels synchronized ...\n", DK_NUM);

	CHECK(cudaDeviceReset());


	// Copy value from output to keys var
	copyValueFromGlobalMemoryToCPUMemory(out->keys, (uint8_t*)output, DK_NUM, DK_LEN, THREAD_X_KERNEL * H_LEN);


	// Debug print
	if(DEBUG) printAllKeys(out->keys, DK_LEN, DK_NUM);


}


__host__ void execution2(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out){

	cudaDeviceProp cudaDeviceProp;
	cudaGetDeviceProperties(&cudaDeviceProp, DEV);

	printf("cudaDeviceProp.deviceOverlap: %d\n", cudaDeviceProp.deviceOverlap);
	assert(cudaDeviceProp.deviceOverlap != 0);

	//Alloc and init CPU memory
	int const N_BYTES_OUTPUT =  THREAD_X_KERNEL * H_LEN * DK_NUM * sizeof(char);

	char	 *output;
	CHECK(cudaMallocHost((void**)&output, N_BYTES_OUTPUT));
	memset(output, 0, N_BYTES_OUTPUT);

	printf("N_BYTES_OUTPUT: %s Bytes\n", prettyPrintNumber(N_BYTES_OUTPUT));

	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT, 0, 0);


	//Device var
	char *d_output;
	char *d_sk;
	int *d_kernelId;

	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);


	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY
	cudaMalloc((void**)&d_kernelId, DK_NUM*sizeof(int));

	// Output var
	CHECK(cudaMalloc((void**)&d_output, N_BYTES_OUTPUT));
	CHECK(cudaMemset(d_output, 0, N_BYTES_OUTPUT));

	// Source Key
	int N_BYTES_SK = (strlen(SOURCE_KEY) + 1) * sizeof(char); // +1 because of null end char
	CHECK(cudaMalloc((void**)&d_sk, N_BYTES_SK));
	CHECK(cudaMemcpy(d_sk, SOURCE_KEY, N_BYTES_SK, cudaMemcpyHostToDevice));

	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);


	cudaStream_t stream[DK_NUM];

	for(int i = 0; i<DK_NUM; i++){
		CHECK(cudaStreamCreate(&stream[i]));
	}

	//Starting kernel
	printf("Starting %d kernels with stream with %s threads each (%d threads needed)...\n", DK_NUM, prettyPrintNumber(block.x*grid.x), THREAD_X_KERNEL);
	int index;
	double start = seconds();
	for(int i = 0; i < DK_NUM; i++){
		index = i * THREAD_X_KERNEL * H_LEN;
		printKernelDebugInfo(i, THREAD_X_KERNEL, THREAD_X_KERNEL*H_LEN, DK_LEN);

		CHECK(cudaMemcpyAsync(&d_kernelId[i], &i, sizeof(int), cudaMemcpyHostToDevice, stream[i]));
		pbkdf2_2<<<grid, block, 0, stream[i]>>>(d_sk, &d_output[index], &d_kernelId[i]);
		CHECK(cudaMemcpyAsync(&output[index], &d_output[index], DK_LEN * sizeof(char), cudaMemcpyDeviceToHost, stream[i]));

		printf("Copy %d° key, %d Bytes starting index output[%d]\n\n", i+1, DK_LEN*sizeof(char), index);
	}

	for(int i = 0; i<DK_NUM; i++){
		CHECK(cudaStreamSynchronize(stream[i]));
	}

	out->elapsedKernel = seconds() - start;
	printf("%d stream synchronized ...\n", DK_NUM);

	for(int i = 0; i<DK_NUM; i++){
		CHECK(cudaStreamDestroy(stream[i]));
	}

	printf("%d stream destroyed ...\n", DK_NUM);

	// Copy value from output to keys var
	copyValueFromGlobalMemoryToCPUMemory(out->keys, (uint8_t*)output, DK_NUM, DK_LEN, THREAD_X_KERNEL * H_LEN);

	// Debug print
	if(DEBUG) printAllKeys(out->keys, DK_LEN, DK_NUM);

	CHECK(cudaDeviceReset());

}

__host__ void execution3(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out){


	int const N_BYTES_OUTPUT =  THREAD_X_KERNEL * H_LEN * sizeof(char);
	char	 *output = (char*)malloc(N_BYTES_OUTPUT);
	memset(output, 0, N_BYTES_OUTPUT);

	printf("N_BYTES_OUTPUT: %s Bytes\n", prettyPrintNumber(N_BYTES_OUTPUT));

	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT, 0, 0);

	//Device var
	char *d_output;
	char *d_sk;


	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);


	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY

	// Output var
	CHECK(cudaMalloc((void**)&d_output, N_BYTES_OUTPUT));
	CHECK(cudaMemset(d_output, 0, N_BYTES_OUTPUT));

	// Source Key
	int N_BYTES_SK = (strlen(SOURCE_KEY) + 1) * sizeof(char); // +1 because of null end char
	CHECK(cudaMalloc((void**)&d_sk, N_BYTES_SK));
	CHECK(cudaMemcpy(d_sk, SOURCE_KEY, N_BYTES_SK, cudaMemcpyHostToDevice));

	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

	//Starting kernel
	printf("Starting ONE kernels with %s threads (%d threads needed)...\n", prettyPrintNumber(block.x * grid.x), THREAD_X_KERNEL);
	double start = seconds();

	pbkdf2_3<<<grid, block>>>(d_sk, d_output);
	CHECK(cudaMemcpy(output, d_output, N_BYTES_OUTPUT, cudaMemcpyDeviceToHost));

	printf("Copy the all output of %d Bytes compose of %d blocks of %d Bytes\n\n", N_BYTES_OUTPUT, N_BYTES_OUTPUT/H_LEN, H_LEN);

	out->elapsedKernel = seconds() - start;
	printf("Kernel synchronized ...\n", DK_NUM);

	CHECK(cudaDeviceReset());


	// Copy value from output to keys var
	copyValueFromGlobalMemoryToCPUMemory(out->keys, (uint8_t*)output, DK_NUM, DK_LEN, DK_LEN);

	// Debug print
	if(DEBUG) printAllKeys(out->keys, DK_LEN, DK_NUM);

}

__host__ void execution4(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out, int const N_STREAM, int const INDEX){


	cudaDeviceProp cudaDeviceProp;
	cudaGetDeviceProperties(&cudaDeviceProp, DEV);
	assert(cudaDeviceProp.deviceOverlap != 0);

	int const N_BYTES_OUTPUT = THREAD_X_KERNEL * H_LEN * N_STREAM * sizeof(char);
	char	 *output;
	cudaMallocHost((void**)&output, N_BYTES_OUTPUT);
	memset(output, 0, N_BYTES_OUTPUT);

	printf("N_BYTES_OUTPUT: %s Bytes\n", prettyPrintNumber(N_BYTES_OUTPUT));

	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT, 0, 0);

	//Device var
	char *d_output;
	char *d_sk;
	int *d_kernelId;

	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);


	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY
	cudaMalloc((void**)&d_kernelId, N_STREAM*sizeof(int));

	// Output var
	CHECK(cudaMalloc((void**)&d_output, N_BYTES_OUTPUT));
	CHECK(cudaMemset(d_output, 0, N_BYTES_OUTPUT));

	// Source Key
	int N_BYTES_SK = (strlen(SOURCE_KEY) + 1) * sizeof(char); // +1 because of null end char
	CHECK(cudaMalloc((void**)&d_sk, N_BYTES_SK));
	CHECK(cudaMemcpy(d_sk, SOURCE_KEY, N_BYTES_SK, cudaMemcpyHostToDevice));

	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);



	cudaStream_t stream[N_STREAM];

	for(int i = 0; i < N_STREAM; i++){
		cudaStreamCreate(&stream[i]);
	}

	//Starting kernel
	printf("Starting %d kernel with stream with %s threads each (%d threads needed)...\n", N_STREAM, prettyPrintNumber(grid.x * block.x), THREAD_X_KERNEL);

	int nBytes;
	int oIndex;
	double start = seconds();
	for(int i = 0; i < N_STREAM; i++){
		nBytes = THREAD_X_KERNEL * H_LEN;
		oIndex = i*nBytes;
		printKernelDebugInfo(i, THREAD_X_KERNEL, nBytes, DK_LEN);

		CHECK(cudaMemcpyAsync(&d_kernelId[i], &i, sizeof(int), cudaMemcpyHostToDevice, stream[i]));
		pbkdf2_4<<<grid, block>>>(d_sk, &d_output[oIndex], &d_kernelId[i]);
		CHECK(cudaMemcpyAsync(&output[oIndex], &d_output[oIndex], nBytes * sizeof(char), cudaMemcpyDeviceToHost, stream[i]));

		printf("Copy %d° macro-block of %d Bytes, starting at index output[%d]\n\n", i+1, nBytes, oIndex);
	}

	for(int i = 0; i < N_STREAM; i++){
		cudaStreamSynchronize(stream[i]);
	}

	out[INDEX].elapsedKernel = seconds() - start;
	printf("%d stream synchronized...\n", N_STREAM);


	for(int i = 0; i < N_STREAM; i++){
		cudaStreamDestroy(stream[i]);
	}
	printf("%d stream destroyed...\n", N_STREAM);


	// Copy value from output to keys var
	copyValueFromGlobalMemoryToCPUMemory(out[INDEX].keys, (uint8_t*)output, DK_NUM, DK_LEN, DK_LEN);

	CHECK(cudaDeviceReset());

	// Debug print
	if(DEBUG) printAllKeys(out[INDEX].keys, DK_LEN, DK_NUM);
}

__host__ void executionSequential(const char* SOURCE_KEY, int const C, int const DK_LEN, int const DK_NUM, struct Data *out){


	printf("Chiavi: %d\nBlocchi: %d\nIterazioni: %d\n", DK_NUM, DK_LEN/H_LEN, C);

	uint8_t tmp[H_LEN];
	uint8_t output[DK_LEN*DK_NUM];
	for(int numKey = 0; numKey < DK_NUM; numKey++){
		for(int block = 0; block < intDivCeil(DK_LEN , H_LEN); block++){
			for(int iteration = 0; iteration < C; iteration++){
				/**
				 * Predosky, you're up !!!!
				 */

				//FOO GENERATOR: predosky delete it.
				for(int i = 0; i < H_LEN; i++){
					tmp[i] = i+1;
				}
			}
			memcpy(&output[numKey*H_LEN], tmp, H_LEN * sizeof(uint8_t));
			memset(tmp, 0, H_LEN*sizeof(uint8_t));
		}

	}

	//out->key is a linear matrix
	memcpy(out->keys, output, DK_LEN * DK_NUM * sizeof(uint8_t));
}


__host__ void copyValueFromGlobalMemoryToCPUMemory(uint8_t *keys, uint8_t *output, int const NUM, int const LEN, int const OFFSET){
	for(int i = 0, j = 0; i < NUM; i++, j += OFFSET){
		memcpy(&keys[i*LEN], &output[j], LEN);
	}
}

__host__ void printAllKeys(uint8_t *keys, int const LEN, int const NUM){
	int index;
	for(int i=0; i<NUM; i++){
		printf("(%d° key): ", i);
		for(int j=0; j<LEN; j++){
			index = (i * LEN) + j;
			printf("%02x ", keys[index]);
		}
		printf("\n\n");
	}
}

__host__ void printHeader(int const DK_NUM, int const DK_LEN, int const  BX){
	printf("\n- - - - REQUEST - - - - -  \n");
	printf("| %d Keys.\t\t |\n", DK_NUM);
	printf("| %d Bytes per Key.\t |\n", DK_LEN);
	printf("| %d Threads per block.\t |\n", BX);
	printf("| %d Byte H_LEN. \t |\n", H_LEN);
	printf("- - - - - - - - - - - - - \n\n");
}

__host__ void printKernelDebugInfo(int const K_ID, int const THREAD_X_K, int const K_BYTES_GENERATED, int const DK_LEN){
	printf("%d° kernel, %d thread, generate %d Bytes (%d Bytes each block), derived key len %d\n", K_ID+1, THREAD_X_K, K_BYTES_GENERATED, H_LEN, DK_LEN);
}
