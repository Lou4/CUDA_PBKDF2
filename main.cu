#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <curand_kernel.h>
#include "my_C_lib/utils.h"
#include "my_C_lib/CPU_time.h"
#include "hashlib/hmac-sha1.cuh"
#include "hashlib/sha1.cuh"
#include "seqlib/seq_hmac_sha1.cuh"


#define H_LEN 20 // Length in Bytes of the PRF functions' output
#define DEV 0
#define intDivCeil(n, d) ((n + d - 1) / d)
#define SK_MAX_LEN 100

int PRINT_KEY, INFO, SLOW_EXE;

__constant__ char D_SK[SK_MAX_LEN];
__constant__ int D_SK_LEN;
__constant__ long D_DK_LEN;
__constant__ int D_C;
__constant__ int D_N;


__device__ void actualFunction(char* output, int const KERNEL_ID, curandState *randomStates){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= D_N)
		return;

	globalChars globalChars;
	uint8_t salt[H_LEN] = "salt\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
	curandState curandState;

	int saltLen = 4 + sizeof(float);
	int *ptr = (int*)&salt[4];
	long seed = idx;

	uint8_t acc[H_LEN];
	uint8_t buffer[H_LEN];

	curand_init(seed, KERNEL_ID,  0, &curandState);

	//attatching salt
	float rr = curand_uniform(&curandState);

	/* DEBUG
	printf("(%d, %d): %f\n", idx, KERNEL_ID, rr);
	*/

	cudaMemcpyDevice(ptr, &rr, sizeof(float));
	cudaMemcpyDevice(&ptr[1], &D_DK_LEN, sizeof(int));

	if(idx == 0 && KERNEL_ID == 0){
		// BYTES ARE STORED IN BIG ENDIAND
		/*
		uint8_t *foo = (uint8_t*)&rr;
		uint8_t *bar = (uint8_t*)&D_DK_LEN;
		for(int i = 0; i<sizeof(float); i++){
			printf("%02x ", foo[i]);
		}
		printf("\n");
		for(int i = 0; i<sizeof(long); i++){
			printf("%02x ", bar[i]);
		}
		printf("\n");*/
		printf("Salt: ");
		for(int i = 0; i < H_LEN; i++){
			printf("%02x ", salt[i]);
		}
		printf("\n");
	}

	hmac_sha1(D_SK, D_SK_LEN, salt, saltLen, buffer, &globalChars);
	cudaMemcpyDevice(salt, buffer, H_LEN);
	cudaMemcpyDevice(acc, buffer, H_LEN);
	for(int i = 0; i < D_C; i++){
		hmac_sha1(D_SK, D_SK_LEN, salt, H_LEN, buffer, &globalChars);
		cudaMemcpyDevice(salt, buffer, H_LEN);

		for(int i = 0; i < H_LEN; i++){
			acc[i] ^= buffer[i];
		}
	}

	int index;
	for(int i = 0; i < H_LEN; i++){
		index = idx * H_LEN + i;
		output[index] = acc[i];
	}

}

__global__ void pbkdf2(char* output, int *kernelId, curandState *randomStates){
	actualFunction(output, *kernelId, randomStates);
}


__global__ void pbkdf2_2(char* output, int *kernelId, curandState *randomStates){
	actualFunction(output, *kernelId, randomStates);
}

__global__ void pbkdf2_3(char* output, curandState *randomStates){
	actualFunction(output, 0, randomStates);
}

__global__ void pbkdf2_4(char* output, int *kernelId, curandState *randomStates){
	actualFunction(output, *kernelId, randomStates);
}

__host__ void execution1(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out);
__host__ void execution2(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out);
__host__ void execution3(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out);
__host__ void execution4(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out, int const nStream, int const INDEX);
__host__ void executionSequential(const char* SOURCE_KEY, int const C, long const DK_LEN, long const DK_NUM, struct Data *out);
__host__ void copyValueFromGlobalMemoryToCPUMemory(uint8_t *keys, uint8_t *output, int const NUM, int const LEN, int const OFFSET);
__host__ void printAllKeys(uint8_t *keys, int const LEN, int const NUM);
__host__ void printHeader(long const DK_NUM, long const DK_LEN, int const  BX, int const C);
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

	if(c != 9){
		printf("Error !!\n");
		printf("./Project_GPU <Bx> <source_key> <iterations> <len_derived_keys> <num_derived_keys> <PRINT_KEY> <INFO> <SLOW_EXECUTION>\n");

		exit(EXIT_FAILURE);
	}

	printf("- - - - - - - - REMINDER - - - - - - - - \n");
	printf("|    sizeof(curandState): %d Bytes     |\n", sizeof(curandState));
	printf("- - - - - - - - - - - - - - - - - - - - ");

	printf("\n\n\n\n");


	//Host var
	int const BX = atoi(v[1]);				// Thread per block
	char const *SOURCE_KEY = v[2];			// Password
	int const SK_LEN = strlen(SOURCE_KEY);	// Password len
	int const C = atoi(v[3]);				// Number of iteration
	long const DK_LEN = atoi(v[4]);			// Derived Keys' length
	long const DK_NUM = atoi(v[5]);			// Number of derived keys we'll generate
	PRINT_KEY = atoi(v[6]);
	INFO = atoi(v[7]);
	SLOW_EXE = atoi(v[8]);

	int foo;

	assert(isPowOfTwo(BX) == 1);
	assert(isPowOfTwo(DK_LEN) == 1);
	assert(isPowOfTwo(DK_NUM) == 1);
	assert(SK_LEN <= SK_MAX_LEN);

	// One kernel generate one dk, one thread generate one Ti
	int *threadsPerKernel;
	threadsPerKernel = (int*) malloc(sizeof(int));
	*threadsPerKernel = intDivCeil(DK_LEN, H_LEN);		// Threads needed
	int Gx = intDivCeil(*threadsPerKernel, BX);			// Calculate Gx

	//Output var
	int const N_STREAM[] = {2, 4, 8, 16};
	int const S_LEN = 4;
	struct Data *out1, *out2, *out3, *out4, *outS;
	out1 = (struct Data*)malloc(sizeof(struct Data));
	out2 = (struct Data*)malloc(sizeof(struct Data));
	out3 = (struct Data*)malloc(sizeof(struct Data));
	out4 = (struct Data*)malloc(sizeof(struct Data) * S_LEN);
	outS = (struct Data*)malloc(sizeof(struct Data));

	out1->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));
	out2->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));
	out3->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));
	for(int i = 0; i < S_LEN; i++)
		out4[i].keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));
	outS->keys = (uint8_t*)malloc(DK_NUM * DK_LEN * sizeof(uint8_t*));

	printHeader(DK_NUM, DK_LEN, BX, C);

	CHECK(cudaSetDevice(DEV));

	//Tranfer to CONSTANT MEMORY
	int N_BYTES_SK = (strlen(SOURCE_KEY) + 1) * sizeof(char); // +1 because of null end char
	CHECK(cudaMemcpyToSymbol(D_SK, 		SOURCE_KEY, 			N_BYTES_SK));	// Source Key
	CHECK(cudaMemcpyToSymbol(D_SK_LEN, 	&SK_LEN, 			sizeof(int)));	// Source key len
	CHECK(cudaMemcpyToSymbol(D_DK_LEN, 	&DK_LEN, 			sizeof(long)));	// Derived key len
	CHECK(cudaMemcpyToSymbol(D_C, 		&C, 					sizeof(int)));	// Iteration
	CHECK(cudaMemcpyToSymbol(D_N, 		threadsPerKernel, 	sizeof(int)));	// Thread per kernel


	// Without Stream
	printf("- - - - - - Execution one, more kernel no stream - - - - - -\n");
	printf("\nKernel: %ld, Thread per Kernel: %s\n\n", DK_NUM, prettyPrintNumber(*threadsPerKernel));
	double start = seconds();
	if(SLOW_EXE) execution1(DK_LEN, DK_NUM, Gx, BX, *threadsPerKernel, out1);
	out1->elapsedGlobal = seconds() - start;
	printf("- - - - - - - End execution one - - - - - - - - - - - - - -\n");

	printf("\n\n\n\n\n\n\n\n\n* * * **************************************************************************************** * * *\n\n\n\n\n\n\n\n\n");

	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX, C);

	// With Stream
	printf("- - - - - - Execution two, with stream - - - - - -\n");
	printf("\nStream: %ld, Thread per Stream: %s\n\n", DK_NUM, prettyPrintNumber(*threadsPerKernel));
	start = seconds();
	if(SLOW_EXE) execution2(DK_LEN, DK_NUM, Gx, BX, *threadsPerKernel, out2);
	out2->elapsedGlobal = seconds() - start;
	printf("- - - - - - - - End execution two - - - - - - - - \n");


	printf("\n\n\n\n\n\n\n\n\n* * * **************************************************************************************** * * *\n\n\n\n\n\n\n\n\n");

	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX, C);

	// One kernel generate ALL dk, one thread generate one Ti
	*threadsPerKernel = intDivCeil((DK_LEN * DK_NUM), H_LEN);		// Threads needed
	Gx = intDivCeil(*threadsPerKernel, BX);						// Calculate Gx

	//Tranfer to CONSTANT MEMORY
	CHECK(cudaMemcpyToSymbol(D_N, threadsPerKernel, sizeof(int)));	// Thread per kernel

	printf("- - - - - - Execution three, one kernel no stream - - - - - -\n");
	printf("\nKernel: 1, Thread per Kernel: %s\n\n", prettyPrintNumber(*threadsPerKernel));
	start = seconds();
	execution3(DK_LEN, DK_NUM, Gx, BX, *threadsPerKernel, out3);
	out3->elapsedGlobal = seconds() - start;
	printf("- - - - - - - End execution three - - - - - - - - - - - - - -\n");


	printf("\n\n\n\n\n\n\n\n\n* * * **************************************************************************************** * * *\n\n\n\n\n\n\n\n\n");

	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX, C);

	printf("- - - - - - Execution four, rational use of stream - - - - - -\n");
	for(int i = 0; i < S_LEN; i++){
		*threadsPerKernel = intDivCeil((DK_LEN * DK_NUM), (H_LEN*N_STREAM[i]));	// Threads needed

		//Tranfer to CONSTANT MEMORY
		CHECK(cudaMemcpyToSymbol(D_N, threadsPerKernel, sizeof(int)));	// Thread per kernel

		Gx = intDivCeil(*threadsPerKernel, BX);									// Calculate Gx
		printf("\nStream: %d, Threads per Stream: %s\n", N_STREAM[i], prettyPrintNumber(*threadsPerKernel));
		printf("Every stream generate %s Bytes.\n\n", prettyPrintNumber(*threadsPerKernel * H_LEN));


		start = seconds();
		if(SLOW_EXE) execution4(DK_LEN, DK_NUM, Gx, BX, *threadsPerKernel, out4, N_STREAM[i], i);
		out4[i].elapsedGlobal = seconds() - start;
		printf("\n\n\n\n");
	}
	printf("- - - - - - - End execution four - - - - - - - - - - - - - -\n");

	printf("\n\n\n\n\n\n\n\n\n* * * **************************************************************************************** * * *\n\n\n\n\n\n\n\n\n");

	printf("Press enter to continue . . .");
	//scanf("%d", &foo);

	printHeader(DK_NUM, DK_LEN, BX, C);

	printf("- - - - - - Last but not least execution, SEQUENTIAL - - - - - -\n");
	start = seconds();
	executionSequential(SOURCE_KEY, C, DK_LEN, DK_NUM, outS);
	outS->elapsedGlobal = seconds() - start;
	printf("- - - - - - - - End last execution - - - - - - - - - - - - - - -\n");

	printf("\n\n\n\n\n\n\n\n\n* * * **************************************************************************************** * * *\n\n\n\n\n\n\n\n\n");

	printf("Press enter to continue . . .");
	//scanf("%d", &foo);


	// Check correctness first two execution
	printf("check correctness . . .\n");
	for(int i=0; i<DK_NUM; i++){
		if (INFO) printf("check %d key (len %d Bytes)...", i, sizeof(uint8_t)*DK_LEN);
		assert(memcmp(out1->keys, out2->keys, DK_NUM * DK_LEN * sizeof(uint8_t)) == 0);
		if (INFO) printf("ok\n");
	}

	printf("\n\n\n- - - - - - - - - - RESULT - - - - - - - - - - - - - - \n");
	printf("  One kernel per key takes \t %f seconds\n", out1->elapsedGlobal);
	printf("  One stream per key takes \t %f seconds\n", out2->elapsedGlobal);
	printf("  One kernel takes \t %f seconds\n", out3->elapsedGlobal);
	for(int i = 0; i < S_LEN; i++){
		printf("  %d Stream takes \t %f seconds\n", N_STREAM[i], out4[i].elapsedGlobal);
	}
	printf("  Sequential takes \t %f seconds\n", outS->elapsedGlobal);
	printf("\n");


	printf("  One kernel vs One kernel per key: \t %c %2lf\n", 37, 100-((100*out3->elapsedGlobal)/out1->elapsedGlobal));
	printf("  One kernel vs One stream per key: \t %c %2lf\n", 37, 100-((100*out3->elapsedGlobal)/out2->elapsedGlobal));
	for(int i = 0; i < S_LEN; i++){
		printf("  One kernel vs %d stream: \t %c %2lf\n", N_STREAM[i], 37, 100-((100*out3->elapsedGlobal)/out4[i].elapsedGlobal));
	}
	printf("  One kernel vs Sequential: \t %c %2lf\n", 37, 100-((100*out3->elapsedGlobal)/outS->elapsedGlobal));
	printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

	cudaDeviceReset();

	return 0;
}



__host__ void execution1(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out){

	//Alloc and init CPU memory
	long const N_BYTES_OUTPUT =  (long)THREAD_X_KERNEL * H_LEN * DK_NUM * sizeof(char);
	long const N_BYTES_CURAND_STATE = (long)THREAD_X_KERNEL * sizeof(curandState);

	printf("Global memory required: %s Bytes (%s Bytes for keys + %s Bytes for curandState)\n", prettyPrintNumber(N_BYTES_OUTPUT + N_BYTES_CURAND_STATE), prettyPrintNumber(N_BYTES_OUTPUT), prettyPrintNumber(N_BYTES_CURAND_STATE));
	printf("Total length of the keys: %s Bytes (overhead %s Bytes)\n", prettyPrintNumber(DK_LEN * DK_NUM), prettyPrintNumber(N_BYTES_OUTPUT - (DK_LEN * DK_NUM)));
	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT + N_BYTES_CURAND_STATE, 0, INFO);

	//Device var
	char *d_output;
	int *d_kernelId;
	curandState *d_randomStates;

	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);

	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY
	CHECK(cudaMalloc((void**)&d_kernelId, sizeof(int)));
	CHECK(cudaMalloc((void**)&d_output, N_BYTES_OUTPUT));
	CHECK(cudaMalloc((void**)&d_randomStates, N_BYTES_CURAND_STATE));
	CHECK(cudaMemset(d_output, 0, N_BYTES_OUTPUT));
	CHECK(cudaMemset(d_kernelId, 0, sizeof(int)));


	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

	//Starting kernel
	long index;
	double start = seconds();
	time_t t;
	struct tm *timestamp;
	for(int i = 0; i < DK_NUM; i++){
		index = (long)i * THREAD_X_KERNEL * H_LEN;
		if(INFO) printKernelDebugInfo(i, THREAD_X_KERNEL, THREAD_X_KERNEL*H_LEN, DK_LEN);

		CHECK(cudaMemcpy(d_kernelId, &i, sizeof(int), cudaMemcpyHostToDevice));
		pbkdf2<<<grid, block>>>(&d_output[index], d_kernelId, d_randomStates);
		CHECK(cudaMemcpy(&out->keys[i*DK_LEN], &d_output[index], DK_LEN * sizeof(char), cudaMemcpyDeviceToHost));

		// Completness
		t = time(NULL);
		timestamp = gmtime(&t);
		printf("%d° kernel of %d done . . . [%dh %dmin %dsec UTC]\n", i+1, DK_NUM, timestamp->tm_hour, timestamp->tm_min, timestamp->tm_sec);

		if(INFO) printf("Copy %d° key, %d Bytes starting index output[%ld]\n\n", i+1, DK_LEN*sizeof(char), index);
	}

	CHECK(cudaDeviceSynchronize());
	out->elapsedKernel = seconds() - start;

	printf("\n");

	if(INFO) printf("%d kernels synchronized ...\n", DK_NUM);
	if(PRINT_KEY) printAllKeys(out->keys, DK_LEN, DK_NUM);

	CHECK(cudaFree(d_output));
	CHECK(cudaFree(d_kernelId));
	CHECK(cudaFree(d_randomStates));
}

__host__ void execution2(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out){

	cudaDeviceProp cudaDeviceProp;
	cudaGetDeviceProperties(&cudaDeviceProp, DEV);

	if(INFO) printf("cudaDeviceProp.deviceOverlap: %d\n", cudaDeviceProp.deviceOverlap);
	assert(cudaDeviceProp.deviceOverlap != 0);

	//Alloc and init CPU memory
	char	 *output;
	int *kid;
	long const N_BYTES_OUTPUT =  (long)THREAD_X_KERNEL * H_LEN * DK_NUM * sizeof(char);
	long const N_BYTES_CURAND_STATE = (long)THREAD_X_KERNEL * sizeof(curandState);
	int const N_BYTES_KID = DK_NUM * sizeof(int);
	CHECK(cudaMallocHost((void**)&output, N_BYTES_OUTPUT));
	CHECK(cudaMallocHost((void**)&kid, N_BYTES_KID));
	memset(output, 0, N_BYTES_OUTPUT);
	memset(kid, 0, N_BYTES_KID);

	printf("Global memory required: %s Bytes (%s Bytes for keys + %s Bytes for curandState + %s Bytes for kernel IDs)\n", prettyPrintNumber(N_BYTES_OUTPUT + N_BYTES_CURAND_STATE + N_BYTES_KID), prettyPrintNumber(N_BYTES_OUTPUT), prettyPrintNumber(N_BYTES_CURAND_STATE), prettyPrintNumber(N_BYTES_KID));
	printf("Total length of the keys: %s Bytes (overhead %s Bytes)\n", prettyPrintNumber(DK_LEN * DK_NUM), prettyPrintNumber(N_BYTES_OUTPUT - (DK_LEN * DK_NUM)));
	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT + N_BYTES_CURAND_STATE + N_BYTES_KID, 0, INFO);


	//Device var
	char *d_output;
	int *d_kernelId;
	curandState *randomStates;

	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);


	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY
	CHECK(cudaMalloc((void**)&d_kernelId, 	N_BYTES_KID));
	CHECK(cudaMalloc((void**)&d_output, 		N_BYTES_OUTPUT));
	CHECK(cudaMalloc((void**)&randomStates, 	N_BYTES_CURAND_STATE));
	CHECK(cudaMemset(d_kernelId,		0, 		N_BYTES_KID));
	CHECK(cudaMemset(d_output, 		0,		N_BYTES_OUTPUT));


	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

	cudaStream_t stream[DK_NUM];
	cudaEvent_t event[DK_NUM];
	for(int i = 0; i<DK_NUM; i++){
		CHECK(cudaStreamCreate(&stream[i]));
		CHECK(cudaEventCreate(&event[i]));
	}

	//Starting kernel
	long index;
	double start = seconds();
	for(int i = 0; i < DK_NUM; i++){
		index = (long)i * THREAD_X_KERNEL * H_LEN;
		if(INFO) printKernelDebugInfo(i, THREAD_X_KERNEL, THREAD_X_KERNEL*H_LEN, DK_LEN);

		kid[i] = i;

		CHECK(cudaMemcpyAsync(&d_kernelId[i], &kid[i], sizeof(int), cudaMemcpyHostToDevice, stream[i]));
		pbkdf2_2<<<grid, block, 0, stream[i]>>>(&d_output[index], &d_kernelId[i], randomStates);
		CHECK(cudaMemcpyAsync(&output[index], &d_output[index], DK_LEN * sizeof(char), cudaMemcpyDeviceToHost, stream[i]));
		CHECK(cudaEventRecord(event[i], stream[i]));

		if(INFO) printf("Copy %d° key, %d Bytes starting index output[%ld]\n\n", i+1, DK_LEN*sizeof(char), index);
	}

	// Sync and profiling
	time_t t;
	struct tm *timestamp;
	for(int i = 0; i<DK_NUM; i++){
		CHECK(cudaEventSynchronize(event[i]));
		t = time(NULL);
		timestamp = gmtime(&t);
		printf("Stream %d complete . . . [%dh %dmin %dsec UTC]\n", i, timestamp->tm_hour, timestamp->tm_min, timestamp->tm_sec);
	}
	out->elapsedKernel = seconds() - start;

	if(INFO) printf("%d stream synchronized ...\n", DK_NUM);

	for(int i = 0; i<DK_NUM; i++){
		CHECK(cudaStreamDestroy(stream[i]));
		CHECK(cudaEventDestroy(event[i]));
	}

	if(INFO) printf("%d stream destroyed ...\n", DK_NUM);

	// Copy value from output to keys var
	copyValueFromGlobalMemoryToCPUMemory(out->keys, (uint8_t*)output, DK_NUM, DK_LEN, THREAD_X_KERNEL * H_LEN);

	// Debug print
	if(PRINT_KEY) printAllKeys(out->keys, DK_LEN, DK_NUM);

	CHECK(cudaFreeHost(output));
	CHECK(cudaFreeHost(kid));
	CHECK(cudaFree(d_output));
	CHECK(cudaFree(d_kernelId));
	CHECK(cudaFree(randomStates));

}

__host__ void execution3(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out){

	long const N_BYTES_OUTPUT =  (long)THREAD_X_KERNEL * H_LEN * sizeof(char);
	long const N_BYTES_CURAND_STATE = (long)THREAD_X_KERNEL * sizeof(curandState);

	printf("Global memory required: %s Bytes (%s Bytes for keys + %s Bytes for curandState)\n", prettyPrintNumber(N_BYTES_OUTPUT + N_BYTES_CURAND_STATE), prettyPrintNumber(N_BYTES_OUTPUT), prettyPrintNumber(N_BYTES_CURAND_STATE));
	printf("Total length of the keys: %s Bytes (overhead %s Bytes)\n", prettyPrintNumber(DK_LEN * DK_NUM), prettyPrintNumber(N_BYTES_OUTPUT - (DK_LEN * DK_NUM)));
	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT, 0, INFO);

	//Device var
	char *d_output;
	curandState *randomStates;

	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);


	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY
	CHECK(cudaMalloc((void**)&d_output, N_BYTES_OUTPUT));
	CHECK(cudaMemset(d_output, 0, N_BYTES_OUTPUT));
	CHECK(cudaMalloc((void**)&randomStates, N_BYTES_CURAND_STATE));


	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

	//Starting kernel
	double start = seconds();
	pbkdf2_3<<<grid, block>>>(d_output, randomStates);
	CHECK(cudaMemcpy(out->keys, d_output, DK_LEN * DK_NUM, cudaMemcpyDeviceToHost));

	if(INFO) printf("Copy the all output of %d Bytes compose of %d blocks of %d Bytes\n\n", N_BYTES_OUTPUT, N_BYTES_OUTPUT/H_LEN, H_LEN);

	out->elapsedKernel = seconds() - start;
	if(INFO) printf("Kernel synchronized ...\n", DK_NUM);

	// Debug print
	if(PRINT_KEY) printAllKeys(out->keys, DK_LEN, DK_NUM);

	CHECK(cudaFree(d_output));
	CHECK(cudaFree(randomStates));
}

__host__ void execution4(long const DK_LEN, long const DK_NUM, int const GX, int const BX, int const THREAD_X_KERNEL, struct Data *out, int const N_STREAM, int const INDEX){

	cudaDeviceProp cudaDeviceProp;
	cudaGetDeviceProperties(&cudaDeviceProp, DEV);
	assert(cudaDeviceProp.deviceOverlap != 0);

	long const N_BYTES_OUTPUT = (long)THREAD_X_KERNEL * H_LEN * N_STREAM * sizeof(char);
	long const N_BYTES_CURAND_STATE = (long)THREAD_X_KERNEL * sizeof(curandState);
	int const N_BYTES_KID = N_STREAM * sizeof(int);
	char	 *output;
	int *kid;
	CHECK(cudaMallocHost((void**)&output, N_BYTES_OUTPUT));
	CHECK(cudaMallocHost((void**)&kid, N_BYTES_KID));
	memset(kid, 0, N_BYTES_KID);
	memset(output, 0, N_BYTES_OUTPUT);

	printf("Global memory required: %s Bytes (%s Bytes for keys + %s Bytes for curandState + %s Bytes for kernel IDs)\n", prettyPrintNumber(N_BYTES_OUTPUT + N_BYTES_CURAND_STATE + N_BYTES_KID), prettyPrintNumber(N_BYTES_OUTPUT), prettyPrintNumber(N_BYTES_CURAND_STATE), prettyPrintNumber(N_BYTES_KID));
	printf("Total length of the keys: %s Bytes (overhead %s Bytes)\n", prettyPrintNumber(DK_LEN * DK_NUM), prettyPrintNumber(N_BYTES_OUTPUT - (DK_LEN * DK_NUM)));
	checkArchitecturalBoundaries(DEV, GX, 1, BX, 1, N_BYTES_OUTPUT, 0, INFO);

	//Device var
	char *d_output;
	int *d_kernelId;
	curandState *randomStates;

	dim3 grid(GX, 1, 1);
	dim3 block(BX, 1, 1);


	//- - - ALLOC AND TRANFER TO GLOBAL MEMORY
	CHECK(cudaMalloc((void**)&d_kernelId, 	N_BYTES_KID));
	CHECK(cudaMalloc((void**)&d_output, 		N_BYTES_OUTPUT));
	CHECK(cudaMalloc((void**)&randomStates, 	N_BYTES_CURAND_STATE));
	CHECK(cudaMemset(d_kernelId, 	0, 	N_BYTES_KID));
	CHECK(cudaMemset(d_output, 		0, 	N_BYTES_OUTPUT));



	printf("grid(%d, %d, %d) - block(%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);


	cudaStream_t stream[N_STREAM];
	cudaEvent_t startEvent[N_STREAM];
	cudaEvent_t stopEvent[N_STREAM];
	for(int i = 0; i < N_STREAM; i++){
		CHECK(cudaStreamCreate(&stream[i]));
		CHECK(cudaEventCreate(&startEvent[i]));
		CHECK(cudaEventCreate(&stopEvent[i]));
	}

	//Starting kernel
	long nBytes;
	long oIndex;
	double start = seconds();
	for(int i = 0; i < N_STREAM; i++){
		nBytes = (long)THREAD_X_KERNEL * H_LEN;
		oIndex = nBytes * i;
		if(INFO) printKernelDebugInfo(i, THREAD_X_KERNEL, nBytes, DK_LEN);

		kid[i] = i;

		CHECK(cudaEventRecord(startEvent[i]));
		CHECK(cudaMemcpyAsync(&d_kernelId[i], &kid[i], sizeof(int), cudaMemcpyHostToDevice, stream[i]));
		pbkdf2_4<<<grid, block, 0, stream[i]>>>(&d_output[oIndex], &d_kernelId[i], randomStates);
		CHECK(cudaMemcpyAsync(&output[oIndex], &d_output[oIndex], nBytes * sizeof(char), cudaMemcpyDeviceToHost, stream[i]));
		CHECK(cudaEventRecord(stopEvent[i]));

		if(INFO) printf("Copy %d° macro-block of %d Bytes, starting at index output[%d]\n\n", i+1, nBytes, oIndex);
	}

	// Sync and profiling
	time_t t;
	struct tm *timestamp;
	float elapsed;
	for(int i = 0; i<N_STREAM; i++){
		CHECK(cudaEventSynchronize(stopEvent[i]));
		t = time(NULL);
		timestamp = gmtime(&t);
		cudaEventElapsedTime(&elapsed, startEvent[i], stopEvent[i]);
		printf("Stream %d complete (elapsed %f millisec). . . [%dh %dmin %dsec UTC]\n", i, elapsed, timestamp->tm_hour, timestamp->tm_min, timestamp->tm_sec);
	}
	out[INDEX].elapsedKernel = seconds() - start;

	if(INFO) printf("%d stream synchronized...\n", N_STREAM);

	for(int i = 0; i < N_STREAM; i++){
		CHECK(cudaStreamDestroy(stream[i]));
		CHECK(cudaEventDestroy(startEvent[i]));
		CHECK(cudaEventDestroy(stopEvent[i]));
	}

	if(INFO) printf("%d stream destroyed...\n", N_STREAM);


	// Copy value from output to keys var
	copyValueFromGlobalMemoryToCPUMemory(out[INDEX].keys, (uint8_t*)output, DK_NUM, DK_LEN, DK_LEN);

	// Debug print
	if(PRINT_KEY) printAllKeys(out[INDEX].keys, DK_LEN, DK_NUM);

	CHECK(cudaFreeHost(output));
	CHECK(cudaFreeHost(kid));
	CHECK(cudaFree(d_output));
	CHECK(cudaFree(d_kernelId));
	CHECK(cudaFree(randomStates));
}

__host__ void executionSequential(const char* SOURCE_KEY, int const TOTAL_ITERATIONS, long const DK_LEN, long const DK_NUM, struct Data *out){

	//srand(time(NULL));
	const int NUM_BLOCKS = intDivCeil(DK_LEN, H_LEN);

	printf("Chiavi: %d\nBlocchi: %d\nIterazioni: %d\n", DK_NUM, NUM_BLOCKS, TOTAL_ITERATIONS);

	char salt[H_LEN] = "salt";
	uint8_t buffer[H_LEN];
	uint8_t k_xor[H_LEN];
	int *ptr = (int*) &salt[strlen(salt)];
	const unsigned int sk_len = strlen(SOURCE_KEY);
	const unsigned int salt_len = strlen(salt);

	memset(ptr, 0, H_LEN - strlen(salt));

	if (INFO) {
		printf("Source Key: %s | len : %d\n", SOURCE_KEY, sk_len);
		printf("Total Iterations: %d\n", TOTAL_ITERATIONS);
		printf("Nun Blocks: %d\n", NUM_BLOCKS);
	}

	time_t t;
	struct tm *timestamp;
	int x = 0;
	int total = DK_NUM * NUM_BLOCKS * TOTAL_ITERATIONS;
	int tenPercent = total / 10;

	t = time(NULL);
	timestamp = gmtime(&t);
	if(tenPercent != 0) printf("%c 0 complete . . . [%dh %dmin %dsec UTC]\n", 37, timestamp->tm_hour, timestamp->tm_min, timestamp->tm_sec);
	for(int numKey = 0; numKey < DK_NUM; numKey++) {

		uint8_t acc_key[NUM_BLOCKS * H_LEN];

		for(int block = 0; block < NUM_BLOCKS; block++) {
			//copy the well know salt value
			//memcpy(buffer, salt, salt_len);
			//concatenate values to add entropy to the salt
			ptr[0] = rand();
			memcpy(&ptr[1], &DK_LEN, sizeof(long));

			if (block == 0 && numKey == 0) {
				printf("(key %d) Salt: ", numKey);
				for (int i = 0; i<H_LEN; i++) {
					printf("%02x ", (uint8_t)salt[i]);
				}
				printf("\n");
			}

			//calculate the fist hmac_sha1
			lrad_hmac_sha1((const unsigned char*) SOURCE_KEY, sk_len, (const unsigned char*) salt, H_LEN, buffer);
			//init the xor val
			memcpy(k_xor, buffer, H_LEN);
			//copy back to salt array
			memcpy(salt, buffer, H_LEN);
			//apply iterations to hash fn
			for(int iteration = 0; iteration < TOTAL_ITERATIONS; iteration++) {
				//hash again
				lrad_hmac_sha1((const unsigned char*) SOURCE_KEY, sk_len, (const unsigned char*) salt, H_LEN, buffer);
				//copy back to salt array for the next iteration
				memcpy(salt, buffer, H_LEN);
				//to optimize the algorithm directly xor the sha1 obtained
				for(int k = 0; k < H_LEN; k++) {
					k_xor[k] ^= buffer[k];
				}

				// Completeness
				x++;
				t = time(NULL);
				timestamp = gmtime(&t);
				if(tenPercent != 0 && x % tenPercent  == 0) printf("%c %d complete . . . [%dh %dmin %dsec UTC]\n", 37, 10 * (x / tenPercent), timestamp->tm_hour, timestamp->tm_min, timestamp->tm_sec);
			}
			//concatenate the key part
			memcpy(&acc_key[block * H_LEN], k_xor, H_LEN);
		}
		//save generated key
		memcpy(&out->keys[numKey * DK_LEN], acc_key, DK_LEN);
	}
	printf("x: %d\n",x);

	if(PRINT_KEY) printAllKeys(out->keys, DK_LEN, DK_NUM);

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

__host__ void printHeader(long const DK_NUM, long const DK_LEN, int const  BX, int const C){
	printf("\n- - - - - - - - - REQUEST - - - - - - - -\n");
	printf("| Keys:              %15s \t |\n", prettyPrintNumber(DK_NUM));
	printf("| Bytes per Key:     %15s \t |\n", prettyPrintNumber(DK_LEN));
	printf("| Threads per block: %15d \t |\n", BX);
	printf("| H_LEN Bytes:       %15d \t |\n", H_LEN);
	printf("| Iterations:        %15s \t |\n", prettyPrintNumber(C));
	printf("- - - - - - - - - - - - - - - - - - - - -\n\n");
}

__host__ void printKernelDebugInfo(int const K_ID, int const THREAD_X_K, int const K_BYTES_GENERATED, int const DK_LEN){
	printf("%d° kernel, %d thread, generate %d Bytes (%d Bytes each block), derived key len %d\n", K_ID+1, THREAD_X_K, K_BYTES_GENERATED, H_LEN, DK_LEN);
}
