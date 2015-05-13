#include <stdio.h>
#include "CudaFunc.cu"

#ifdef WIN64
#include <windows.h>
#else
#include <time.h>
#endif

double GetSystemTime(void)
{
#ifdef WIN64
	LARGE_INTEGER Frequency;
	BOOL UseHighPerformanceTimer = QueryPerformanceFrequency(&Frequency);
	// High performance counter available : use it
	LARGE_INTEGER CurrentTime;
	QueryPerformanceCounter(&CurrentTime);
	return static_cast<double>(CurrentTime.QuadPart) / Frequency.QuadPart;
#else
	timespec time;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time);
	return time.tv_sec + time.tv_nsec * 1e-9;
#endif
}

int main()
{
	CudaFunc<char> cudaFunc;



	return 0;
}