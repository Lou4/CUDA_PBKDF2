void print_thread_logic(int const MATRIX_NUM_ROW, int const MATRIX_NUM_COL, int const TOT_ROW_ELEMS, int const TOT_COL_ELEMS, int const THREAD_PER_BLOCK_X, int const THREAD_PER_BLOCK_Y);
void calculate_grid_block_dim(int *gx, int *gy, int *bx, int *by, int const R, int const C);
void print_linear_matrix(int *m, int const R, int const C);
void print_linear_matrix(float *m, int const R, int const C);
int* createInitializedArray(int const LEN, int const RANGE, int const OFFSET);
float* createInitializedArray(int const LEN, int const RANGE, float const OFFSET);
int isPowOfTwo(int num);
char* prettyPrintNumber(long n);
int checkDeviceProperties(int b_x, int b_y, int b_z, int g_x, int g_y, int g_z, int globalMem, int dev);
int checkArchitecturalBoundaries(int const DEV, int const Gx, int const Gy, int const Bx, int const By, int const N_BYTES_GLOBAL_MEMORY_USAGE, int SMEM_REQUIRED, int PRINT);

#define BLOCK_DIM_X 12
#define BLOCK_DIM_Y 10
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}
#define INDEX(x, y, stride) (y * stride + x)
