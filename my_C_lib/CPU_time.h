/*
 * CPU_time.h
 *
 *  Created on: 23/mar/2017
 *      Author: grossi
 */
#include <sys/time.h>

#ifndef CPU_TIME_H_
#define CPU_TIME_H_

inline double seconds() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

#endif /* CPU_TIME_H_ */
