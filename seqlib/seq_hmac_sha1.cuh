#ifndef _SEQ_HMAC_SHA1_H
#define _SEQ_HMAC_SHA1_H

void lrad_hmac_sha1(const unsigned char *text, int text_len, const unsigned char *key,  int key_len, unsigned char *digest);


#endif
