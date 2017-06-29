#include <stdint.h>
#define BLOCK_DIM 32
#define TEST_SIZE 4096

#define SWAP(n) (n >> 24) | ((n << 8) & 0x00FF0000) | ((n >> 8) & 0x0000FF00) | (n << 24)

/* SHA1 round constants */
#define K1 0x5a827999
#define K2 0x6ed9eba1
#define K3 0x8f1bbcdc
#define K4 0xca62c1d6

/* Round functions.  Note that F2 is the same as F4.  */
#define F1(B,C,D) ( D ^ ( B & ( C ^ D ) ) )
#define F2(B,C,D) (B ^ C ^ D)
#define F3(B,C,D) ( ( B & C ) | ( D & ( B | C ) ) )
#define F4(B,C,D) (B ^ C ^ D)

/* Structure to save state of computation between the single steps.  */
struct sha1_ctx
{
  uint32_t A;
  uint32_t B;
  uint32_t C;
  uint32_t D;
  uint32_t E;

  uint32_t total[2];
  uint32_t buflen;
  uint32_t buffer[32];
};


struct globalChars {
  char block[64];
  char innerhash[20];
};

__device__ void cudaMemcpyDevice(void * dst, const void * src, size_t len);
__device__ void cudaMemsetDevice ( void * ptr, uint32_t value, size_t num);
__device__ void sha1_init_ctx (struct sha1_ctx * ctx);
__forceinline__ __device__ void set_uint32 (char *cp, uint32_t v);
__device__ void sha1_read_ctx (const struct sha1_ctx * ctx, void *resbuf);
__device__ void sha1_finish_ctx (struct sha1_ctx *ctx, void *resbuf);
__device__ void sha1_process_bytes (const void *buffer, size_t len, struct sha1_ctx * ctx);
__device__ void sha1_process_block (const void *buffer, size_t len, struct sha1_ctx *ctx);
