#ifndef _HMAC_CUDA_H
#define _HMAC_CUDA_H
#include <stdint.h>
#define IPAD 0x36363636
#define OPAD 0x5c5c5c5c

__device__ void memxor (void * dest, const void * src,size_t n);
__device__ void hmac_sha1 (const void * key, uint32_t keylen, const void *in, uint32_t inlen, void *resbuf, struct globalChars *chars);
#endif
