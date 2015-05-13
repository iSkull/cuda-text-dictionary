
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "reduction_kernel.cu"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#ifdef WIN64
#include <windows.h>
#else
#include <time.h>
#endif

#define gThread	256
#define gBlock	64
#define gKernel	6
#define gLoop	100

// =====================================	HOST FUNCTION	===========================================

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

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

extern "C"
bool isPow2(unsigned int x)
{
	return ((x&(x - 1)) == 0);
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

void remove_buffer(void* ptr, char* property);
void remove_buffer(thrust::device_ptr<void> ptr, char* property);

void compute_block_thread(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

unsigned int compute_sum_kernel(int count, int threads, int blocks, int maxthreads, int maxblocks, int wkernel, unsigned int *input, unsigned int *output);

// =============================================================================================-======

// =====================================	DEVICE FUNCTION	===========================================

__constant__ unsigned int UINT_MASK[32] =
{
	0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010
	, 0x00000020, 0x00000040, 0x00000080, 0x00000100, 0x00000200
	, 0x00000400, 0x00000800, 0x00001000, 0x00002000, 0x00004000
	, 0x00008000, 0x00010000, 0x00020000, 0x00040000, 0x00080000
	, 0x00100000, 0x00200000, 0x00400000, 0x00800000, 0x01000000
	, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000
	, 0x40000000, 0x80000000
};

struct is_true
{
	__host__ __device__
		bool operator()(const bool x)
	{
		return x;
	}
};

struct is_one
{
	__host__ __device__
		bool operator()(const unsigned int x)
	{
		return (x == 1);
	}
};

__global__ void compute_word_count_kernel(char* input, unsigned int* line_pos, unsigned int* pos_len, unsigned int* word_count, int count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < count)
	{
		int tick = 1;
		int _count = 0;

		
		int startPos = (tid == 0) ? 0 : line_pos[tid - 1] + 1;
		int endPos = line_pos[tid];

		char *ptr = input + startPos;

		int lineLength = endPos - startPos;
		if (lineLength > 0)
		{
			while (_count < lineLength)
			{
				if (ptr[_count] == ' ')
				{
					if (_count == 0)
						tick = 0;

					tick++;
					while (ptr[_count + 1] == ' ')
					{
						_count++;
					}
				}

				_count++;
			}
			word_count[tid] = tick;
			pos_len[tid] = lineLength;
		}
		tid += gridDim.x * blockDim.x;
	}
}

__global__ void compute_word_kernel(char* input, unsigned int* ext_line_pos, unsigned int* ext_line_len, unsigned int* ext_word_count, unsigned int* start_pos, unsigned int* end_pos, unsigned int* word_len, unsigned int* line_idx, int count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < count)
	{
		int offset = ext_word_count[tid - 1];
		unsigned int* wl = tid > 0 ? word_len + offset : word_len;
		unsigned int* sp = tid > 0 ? start_pos + offset : start_pos;
		unsigned int* ep = tid > 0 ? end_pos + offset : end_pos;
		unsigned int* li = tid > 0 ? line_idx + offset : line_idx;

		int _count = 0;
		int endPos = ext_line_pos[tid];
		int lineLength = ext_line_len[tid];
		int startPos = endPos - lineLength;

		int startPosCount = 0;
		int endPosCount = 0;
		int wordLenCount = 0;
		int wordSizeCount = 0;

		char *ptr = input + startPos;

		while (_count < lineLength)
		{
			if (ptr[_count] == ' ' && _count > 0)
			{
				ep[endPosCount] = startPos + _count - 1;

				while (ptr[_count + 1] == ' ')
				{
					_count++;
				}

				sp[startPosCount] = startPos + _count + 1;
				wl[wordLenCount] = wordSizeCount;
				li[wordLenCount] = tid;
				
				wordSizeCount = 0;
				wordLenCount++;
				endPosCount++;
				startPosCount++;
				_count++;
			}

			if (ptr[_count] == ' ' && _count == 0)
			{
				while (ptr[_count + 1] == ' ')
				{
					_count++;
				}

				sp[startPosCount] = startPos + _count + 1;
				startPosCount++;
				_count++;
			}

			if (ptr[_count] != ' ' && _count == 0)
			{
				sp[startPosCount] = startPos;
				startPosCount++;
			}

			_count++;
			wordSizeCount++;
		}
		wl[wordLenCount] = wordSizeCount;
		ep[endPosCount] = endPos;
		li[wordLenCount] = tid;
		tid += gridDim.x * blockDim.x;
	}
}

__global__ void compute_word(char* input, unsigned int* start_pos, unsigned int* end_pos, unsigned int* word_len, unsigned int* word_pos, char* output, int count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < count)
	{
		int start = start_pos[tid];
		int end = end_pos[tid];
		int len = word_len[tid];
		int offset = tid > 0 ? word_pos[tid - 1] : 0;

		if (tid > 1)
		{
			offset - 1;
		}

		int _count = 0;

		char* ptr = input + start;

		while (_count < len)
		{
			output[offset + _count] = ptr[_count];
			_count++;
		}

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void compute_line_kernel(char* input, bool* pos, unsigned int* output, int count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < count)
	{
		int inc = 0;
		unsigned int out = 0;
		int index = tid * 32;
		char* ptr = input + index;

		while (inc < 32)
		{
			char _char = ptr[inc];

			//if (_char == '\0')
			//	break;

			if (_char == '\n')
			{
				out |= UINT_MASK[inc];

				pos[index + inc] = true;
			}

			inc++;
		}
		output[tid] = out;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void compute_have_word(unsigned int* input, unsigned int* output, int count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < count)
	{
		int inc = 0;
		unsigned int out = 0;
		unsigned int* ptr = input + tid * 32;
		while (inc < 32)
		{
			if (*ptr > 0)
				out |= UINT_MASK[inc];

			ptr++;
			inc++;
		}
		output[tid] = out;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void compute_have_word_2(unsigned int* input, unsigned int* output, int count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < count)
	{
		if (input[tid] > 0)
			output[tid] = 1;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void compute_bit_count_kernel(unsigned int* var, int count)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < count)
	{
		unsigned int data = var[tid];
		data = data - ((data >> 1) & 0x55555555);
		data = (data & 0x33333333) + ((data >> 2) & 0x33333333);
		var[tid] = (((data + (data >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void MiCuda_ConvertCharToUnsignedInt(
	char	*data,
	unsigned int	*index,
	unsigned int	*dataValues,
	unsigned long long	totalData,
	unsigned int	*wordLength,
	unsigned int	*wordLengthPS,
	unsigned int	charPos
	)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	while (id < totalData)
	{
		unsigned int off = index[id];
		unsigned int adj = charPos - 1;
		unsigned int pos = (off > 0) ? wordLengthPS[off - 1] + adj : adj;
		int diff = (int)charPos - wordLength[off];

		if (diff >= 4)
		{
			dataValues[id] = 0;
		}

		if ((0 < diff) && (diff < 4))
		{
			unsigned int temp = 0;
			if (diff == 1)
			{
				temp = (data[pos - 1] & 0x000000FF | (data[pos - 2] << 8) & 0x0000FF00 | (data[pos - 3] << 16) & 0x00FF0000);
				dataValues[id] = temp << 8;
			}

			if (diff == 2)
			{
				temp = (data[pos - 2] & 0x000000FF | (data[pos - 3] << 8) & 0x0000FF00);
				dataValues[id] = temp << 16;
			}

			if (diff == 3)
			{
				temp = (data[pos - 3] & 0x000000FF);
				dataValues[id] = temp << 24;
			}
		}

		if (diff < 1)
		{
			dataValues[id] = (data[pos] & 0x000000FF |
				(data[pos - 1] << 8) & 0x0000FF00 |
				(data[pos - 2] << 16) & 0x00FF0000 |
				(data[pos - 3] << 24) & 0xFF000000);
		}

		id += blockDim.x * gridDim.x;
	}
}

__global__ void MiCuda_ConvertCharToUnsignedIntFinal(
	char	*data,
	unsigned int	*index,
	unsigned int	*dataValues,
	unsigned long long	totalData,
	unsigned int	*wordLength,
	unsigned int	*wordLengthPS,
	unsigned int	charPos
	)
{
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	while (id < totalData)
	{
		unsigned int off = index[id];
		unsigned int adj = charPos - 1;
		unsigned int pos = (off > 0) ? wordLengthPS[off - 1] + adj : adj;
		int diff = (int)charPos - wordLength[off];

		if (diff < 0)
		{
			unsigned int _data = 0;
			if (charPos == 1) {
				_data = (data[pos] & 0x000000FF);
				dataValues[id] = (_data << 24);
			}

			if (charPos == 2) {
				_data = (data[pos] & 0x000000FF |
					(data[pos - 1] << 8) & 0x0000FF00);
				dataValues[id] = (_data << 16);
			}

			if (charPos == 3) {
				_data = (data[pos] & 0x000000FF |
					(data[pos - 1] << 8) & 0x0000FF00 |
					(data[pos - 2] << 16) & 0x00FF0000);
				dataValues[id] = (_data << 8);
			}

			if (charPos == 4) {
				dataValues[id] = (data[pos] & 0x000000FF |
					(data[pos - 1] << 8) & 0x0000FF00 |
					(data[pos - 2] << 16) & 0x00FF0000 |
					(data[pos - 3] << 24) & 0xFF000000);
			}
		}

		if ((0 <= diff) && (diff < 4))
		{
			unsigned int temp = 0;
			if (diff == 0)
			{
				dataValues[id] = (data[pos] & 0x000000FF | (data[pos - 1] << 8) & 0x0000FF00 | (data[pos - 2] << 16) & 0x00FF0000 | (data[pos - 3] << 24) & 0xFF000000);
			}

			if (diff == 1)
			{
				temp = (data[pos - 1] & 0x000000FF | (data[pos - 2] << 8) & 0x0000FF00 | (data[pos - 3] << 16) & 0x00FF0000);
				dataValues[id] = temp << 8;
			}

			if (diff == 2)
			{
				temp = (data[pos - 2] & 0x000000FF | (data[pos - 3] << 8) & 0x0000FF00);
				dataValues[id] = temp << 16;
			}

			if (diff == 3)
			{
				temp = (data[pos - 3] & 0x000000FF);
				dataValues[id] = temp << 24;
			}
		}

		id += blockDim.x * gridDim.x;
	}
}



__global__ void MiCuda_GatherVarchar(char *inData, char *outData, unsigned int *index, unsigned int maxLength, unsigned long long totalData)
{
	unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;

	while (offset < totalData) {
		for (unsigned int i = 0; i < maxLength; i++) {
			outData[(offset*maxLength) + i] = inData[(index[offset] * maxLength) + i];
		}
		offset += blockDim.x * gridDim.x;
	}
}

__global__ void MiCuda_GatherIndex(unsigned int *index, unsigned int *refIndex, unsigned int *tempIndex, unsigned long long totalData)
{
	unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;

	while (offset < totalData) {
		tempIndex[offset] = refIndex[index[offset]];
		offset += blockDim.x * gridDim.x;
	}
}

__global__ void hash_word(void* key, uint64_t* output, const unsigned int* len, const unsigned int* off, unsigned int totalData)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < totalData)
	{
		const unsigned int length = len[tid];

		const uint64_t m = 0xc6a4a7935bd1e995;
		const int r = 47;
		uint64_t h = 100u ^ (length * m);

		unsigned int offset = tid > 0 ? off[tid - 1] : 0;
		const char* data = (const char*)key + offset;
		const char* end = data + (length / 8 * 8);

		while (data != end)
		{
			uint64_t k = uint64_t(*data++ & 0xFF);
			k |= uint64_t(*data++ & 0xFF) << 8;
			k |= uint64_t(*data++ & 0xFF) << 16;
			k |= uint64_t(*data++ & 0xFF) << 24;
			k |= uint64_t(*data++ & 0xFF) << 32;
			k |= uint64_t(*data++ & 0xFF) << 40;
			k |= uint64_t(*data++ & 0xFF) << 48;
			k |= uint64_t(*data++ & 0xFF) << 56;

			k *= m;
			k ^= k >> r;
			k *= m;

			h ^= k;
			h *= m;
		}

		switch (length & 7)
		{
		case 7: h ^= uint64_t(data[6] & 0xFF) << 48;
		case 6: h ^= uint64_t(data[5] & 0xFF) << 40;
		case 5: h ^= uint64_t(data[4] & 0xFF) << 32;
		case 4: h ^= uint64_t(data[3] & 0xFF) << 24;
		case 3: h ^= uint64_t(data[2] & 0xFF) << 16;
		case 2: h ^= uint64_t(data[1] & 0xFF) << 8;
		case 1: h ^= uint64_t(data[0] & 0xFF);
			h *= m;
		};

		h ^= h >> r;
		h *= m;
		h ^= h >> r;

		output[tid] = h;
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void reshuffle(uint64_t* src, uint64_t* dest, unsigned int* index, unsigned int totalData)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < totalData)
	{
		unsigned int idx = index[tid];
		dest[tid] = src[idx];

		tid += blockDim.x * gridDim.x;
	}
}

// =============================================================================================-======

void host_hash_word(void* key, uint64_t &output, const unsigned int len, unsigned int totalData)
{
	const unsigned int length = len;

	const uint64_t m = 0xc6a4a7935bd1e995;
	const int r = 47;
	uint64_t h = 100u ^ (length * m);

	const char* data = (const char*)key;
	const char* end = data + (length / 8 * 8);

	while (data != end)
	{
		uint64_t k = uint64_t(*data++ & 0xFF);
		k |= uint64_t(*data++ & 0xFF) << 8;
		k |= uint64_t(*data++ & 0xFF) << 16;
		k |= uint64_t(*data++ & 0xFF) << 24;
		k |= uint64_t(*data++ & 0xFF) << 32;
		k |= uint64_t(*data++ & 0xFF) << 40;
		k |= uint64_t(*data++ & 0xFF) << 48;
		k |= uint64_t(*data++ & 0xFF) << 56;

		k *= m;
		k ^= k >> r;
		k *= m;

		h ^= k;
		h *= m;
	}

	switch (length & 7)
	{
	case 7: h ^= uint64_t(data[6] & 0xFF) << 48;
	case 6: h ^= uint64_t(data[5] & 0xFF) << 40;
	case 5: h ^= uint64_t(data[4] & 0xFF) << 32;
	case 4: h ^= uint64_t(data[3] & 0xFF) << 24;
	case 3: h ^= uint64_t(data[2] & 0xFF) << 16;
	case 2: h ^= uint64_t(data[1] & 0xFF) << 8;
	case 1: h ^= uint64_t(data[0] & 0xFF);
		h *= m;
	};

	h ^= h >> r;
	h *= m;
	h ^= h >> r;

	output = h;
}

int main()
{
	int numBlocks = 0;
	int binarySize = 0;
	int numThreads = 0;
	double start = 0.0;
	unsigned int totalRow = 0;
	unsigned int totalWord = 0;
	unsigned int totalHaveWord = 0;
	unsigned int totalWordSize = 0;
	std::string filename = "english.50MB";

	FILE* input = fopen(filename.c_str(), "r");
	if (input)
	{
		printf("%s opened\n", filename.c_str());

		fseek(input, 0L, SEEK_END);
		int textSize = ftell(input);
		fseek(input, 0L, SEEK_SET);
		textSize = textSize;
		printf("File size: %d bytes\n", textSize);

		char* hostBuffer = (char*)malloc(textSize);
		if (hostBuffer)
		{
			memset(hostBuffer, 0, textSize);

			bool* lineStat = NULL;
			unsigned int* linePos = NULL;
			unsigned int* lineCount = NULL;
			unsigned int* wordCount = NULL;
			unsigned int* lineLength = NULL;

			binarySize = textSize / 32;		// size for count
			
			if (textSize % 32 != 0)
				binarySize++;

			// =====================	ALLOC	==========================
			lineCount = (unsigned int*)malloc(binarySize * sizeof(unsigned int));
			lineStat = (bool*)malloc(textSize * sizeof(bool));
			memset(lineCount, 0, binarySize * sizeof(unsigned int));
			memset(lineStat, 0, textSize * sizeof(bool));
			// ===========================================================

			printf("File read: %llu bytes\n", fread(hostBuffer, 1, textSize, input));

			thrust::device_ptr<char> dev_data_word;
			thrust::device_ptr<char> dev_data;
			thrust::device_ptr<bool> dev_data_row_flag;
			thrust::device_ptr<unsigned int> dev_block;
			thrust::device_ptr<unsigned int> dev_data_row_pos;
			thrust::device_ptr<unsigned int> dev_data_word_end;
			thrust::device_ptr<unsigned int> dev_data_word_row;
			thrust::device_ptr<unsigned int> dev_data_word_start;
			thrust::device_ptr<unsigned int> dev_data_row_count;
			thrust::device_ptr<unsigned int> dev_data_word_count;
			thrust::device_ptr<unsigned int> dev_data_row_len;
			thrust::device_ptr<unsigned int> dev_data_word_len;
			thrust::device_ptr<unsigned int> dev_data_word_row_idx;
			thrust::device_ptr<unsigned int> dev_data_row_pos_ext;
			thrust::device_ptr<unsigned int> dev_data_word_count_ext;
			thrust::device_ptr<unsigned int> dev_data_row_len_ext;
			thrust::device_ptr<unsigned int> dev_data_word_len_ps;
			thrust::device_ptr<unsigned int> dev_data_word_count_ext_ps;

			start = GetSystemTime();
			compute_block_thread(gKernel, binarySize, gThread, gBlock, numBlocks, numThreads);
			// =====================	MALLOC	==========================	
			dev_data = thrust::device_malloc<char>(textSize);
			dev_data_row_flag = thrust::device_malloc<bool>(textSize);
			dev_block = thrust::device_malloc<unsigned int>(numBlocks);
			dev_data_row_count = thrust::device_malloc<unsigned int>(binarySize);
			// ===========================================================

			// =====================	MEMSET	==========================
			cudaMemset(dev_data.get(), 0, textSize);
			cudaMemset(dev_data_row_flag.get(), 0, textSize * sizeof(bool));
			cudaMemset(dev_block.get(), 0, numBlocks * sizeof(unsigned int));
			cudaMemset(dev_data_row_count.get(), 0, binarySize * sizeof(unsigned int));
			// ===========================================================

			// =====================	MEMCPY	==========================
			cudaMemcpy(dev_data.get(), hostBuffer, textSize, cudaMemcpyHostToDevice);
			//thrust::copy(hostBuffer, hostBuffer + textSize, dev_data);
			// ===========================================================

			// =====================	KERNEL	==========================
			
			/*----------------- #1 ----------------------*/
			/*	get total row from given input data
				ps: skip this if number of row is known*/
			compute_line_kernel << <gBlock, gThread >> >(dev_data.get(), dev_data_row_flag.get(), dev_data_row_count.get(), binarySize);
			compute_bit_count_kernel << <gBlock, gThread >> >(dev_data_row_count.get(), binarySize);
			totalRow = compute_sum_kernel(binarySize, numThreads, numBlocks, gThread, gBlock, gKernel, dev_data_row_count.get(), dev_block.get());
			/*-------------------------------------------*/

			printf("Total line: %u row\n", totalRow);
			thrust::device_free(dev_data_row_count);

			/*----------------- #2 ----------------------*/
			/*	get total word count from the given input, 
				important step to create new word device buffer for keywords */
			dev_data_word_count = thrust::device_malloc<unsigned int>(totalRow);
			dev_data_row_len = thrust::device_malloc<unsigned int>(totalRow);
			dev_data_row_pos = thrust::device_malloc<unsigned int>(totalRow);
			cudaMemset(dev_data_word_count.get(), 0, totalRow * sizeof(unsigned int));
			cudaMemset(dev_data_row_len.get(), 0, totalRow * sizeof(unsigned int));
			thrust::copy_if(thrust::make_counting_iterator(0u), thrust::make_counting_iterator((unsigned int)textSize), dev_data_row_flag, dev_data_row_pos, is_true());
			compute_word_count_kernel << <gBlock, gThread >> >(dev_data.get(), dev_data_row_pos.get(), dev_data_row_len.get(), dev_data_word_count.get(), totalRow);
			totalWord = compute_sum_kernel(totalRow, numThreads, numBlocks, gThread, gBlock, gKernel, dev_data_word_count.get(), dev_block.get());
			/*-------------------------------------------*/

			printf("Total word: %u row\n", totalWord);
			thrust::device_free(dev_data_row_flag);
			
			/*----------------- #3 ----------------------*/
			/*	get total word count in a row */
			dev_data_word_row = thrust::device_malloc<unsigned int>(totalRow);
			cudaMemset(dev_data_word_row.get(), 0, totalRow * sizeof(unsigned int));
			compute_have_word_2 << <gBlock, gThread >> >(dev_data_word_count.get(), dev_data_word_row.get(), totalRow);
			compute_bit_count_kernel << <gBlock, gThread >> >(dev_data_word_row.get(), totalRow);
			totalHaveWord = compute_sum_kernel(totalRow, numThreads, numBlocks, gThread, gBlock, gKernel, dev_data_word_row.get(), dev_block.get());
			
			/*-------------------------------------------*/

			printf("Total row contain words: %u\n", totalHaveWord);

			/*----------------- #4 ----------------------*/
			/* extract each row length, word count, start position of each row from rows which actually consist words */
			dev_data_row_len_ext = thrust::device_malloc<unsigned int>(totalHaveWord);
			dev_data_word_count_ext = thrust::device_malloc<unsigned int>(totalHaveWord);
			dev_data_row_pos_ext = thrust::device_malloc<unsigned int>(totalHaveWord);
			cudaMemset(dev_data_row_len_ext.get(), 0, totalHaveWord * sizeof(unsigned int));
			cudaMemset(dev_data_word_count_ext.get(), 0, totalHaveWord * sizeof(unsigned int));
			cudaMemset(dev_data_row_pos_ext.get(), 0, totalHaveWord * sizeof(unsigned int));
			thrust::copy_if(dev_data_row_len, dev_data_row_len + totalRow, dev_data_word_row, dev_data_row_len_ext, is_one());
			thrust::copy_if(dev_data_word_count, dev_data_word_count + totalRow, dev_data_word_row, dev_data_word_count_ext, is_one());
			thrust::copy_if(dev_data_row_pos, dev_data_row_pos + totalRow, dev_data_word_row, dev_data_row_pos_ext, is_one());
			
			thrust::device_ptr<unsigned int> dev_data_row_pos_idx = thrust::device_malloc<unsigned int>(totalHaveWord);
			cudaMemset(dev_data_row_pos_idx.get(), 0, totalHaveWord * sizeof(unsigned int));
			thrust::copy_if(thrust::make_counting_iterator(0u), thrust::make_counting_iterator((unsigned int)totalRow), dev_data_word_row,  dev_data_row_pos_idx, is_one());
			/*-------------------------------------------*/

			thrust::device_free(dev_data_word_count);
			//thrust::device_free(dev_data_row_len);
			//thrust::device_free(dev_data_row_pos);
			thrust::device_free(dev_data_word_row);

			/*----------------- #5 ----------------------*/
			/* perform prefix sum on word count per row to get an idea of word offset in word buffer */
			dev_data_word_count_ext_ps = thrust::device_malloc<unsigned int>(totalHaveWord);
			cudaMemset(dev_data_word_count_ext_ps.get(), 0, totalHaveWord * sizeof(unsigned int));
			thrust::inclusive_scan(dev_data_word_count_ext, dev_data_word_count_ext + totalHaveWord, dev_data_word_count_ext_ps);
			/*-------------------------------------------*/

			thrust::device_free(dev_data_word_count_ext);

			/*----------------- #6 ----------------------*/
			/* get each word start position, end position, and word length */
			dev_data_word_start = thrust::device_malloc<unsigned int>(totalWord);
			dev_data_word_end = thrust::device_malloc<unsigned int>(totalWord);
			dev_data_word_len = thrust::device_malloc<unsigned int>(totalWord);
			dev_data_word_row_idx = thrust::device_malloc<unsigned int>(totalWord);
			cudaMemset(dev_data_word_start.get(), 0, totalWord * sizeof(unsigned int));
			cudaMemset(dev_data_word_end.get(), 0, totalWord * sizeof(unsigned int));
			cudaMemset(dev_data_word_len.get(), 0, totalWord * sizeof(unsigned int));
			cudaMemset(dev_data_word_row_idx.get(), 0, totalWord * sizeof(unsigned int));
			compute_word_kernel << <gBlock, gThread >> >(dev_data.get(), dev_data_row_pos_ext.get(), dev_data_row_len_ext.get(), dev_data_word_count_ext_ps.get(), dev_data_word_start.get(), dev_data_word_end.get(), dev_data_word_len.get(), dev_data_word_row_idx.get(), totalHaveWord);
			/*-------------------------------------------*/

			thrust::device_free(dev_data_word_count_ext_ps);
			thrust::device_free(dev_data_row_pos_ext);
			thrust::device_free(dev_data_row_len_ext);

			thrust::device_ptr<unsigned int> deviceMaxLength = thrust::device_malloc<unsigned int>(1);
			deviceMaxLength[0] = thrust::reduce(dev_data_word_len, dev_data_word_len + totalWord, 0, thrust::maximum<unsigned int>());
			unsigned int maxLength[1];
			cudaMemcpy(maxLength, deviceMaxLength.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost);
			printf("Max word length: %u\n", maxLength[0]);

			/*----------------- #7 ----------------------*/
			/* based on the word length, get the total size of all words */
			totalWordSize = compute_sum_kernel(totalWord, numThreads, numBlocks, gThread, gBlock, gKernel, dev_data_word_len.get(), dev_block.get());
			/*-------------------------------------------*/

			printf("Total word size: %u\n", totalWordSize);
			thrust::device_free(dev_block);

			/*----------------- #8 ----------------------*/
			/* copy word from input data to new word device buffer */
			dev_data_word = thrust::device_malloc<char>(totalWordSize);
			dev_data_word_len_ps = thrust::device_malloc<unsigned int>(totalWord);
			cudaMemset(dev_data_word.get(), 0, totalWordSize);
			cudaMemset(dev_data_word_len_ps.get(), 0, totalWord * sizeof(unsigned int));
			thrust::inclusive_scan(dev_data_word_len, dev_data_word_len + totalWord, dev_data_word_len_ps);
			compute_word << <gBlock, gThread >> >(dev_data.get(), dev_data_word_start.get(), dev_data_word_end.get(), dev_data_word_len.get(), dev_data_word_len_ps.get(), dev_data_word.get(), totalWord);
			/*-------------------------------------------*/
			
			thrust::device_free(dev_data);
			thrust::device_free(dev_data_word_start);
			thrust::device_free(dev_data_word_end);

			/*----------------- #8 ----------------------*/
			/* shuffle word index */
			thrust::device_vector<unsigned int> dev_data_word_idx(totalWord);
			thrust::device_vector <unsigned int> dev_temp_idx_output(totalWord);
			thrust::device_vector <unsigned int> dev_temp_idx_input(totalWord);
			thrust::sequence(dev_data_word_idx.begin(), dev_data_word_idx.end());
			thrust::sequence(dev_temp_idx_input.begin(), dev_temp_idx_input.end());
			for (int charPos = maxLength[0]; charPos > 0; charPos = charPos - 4) 
			{
				if (charPos < 5) 
					MiCuda_ConvertCharToUnsignedIntFinal << < 1024, 128 >> > (dev_data_word.get(), dev_temp_idx_input.data().get(), dev_temp_idx_output.data().get(), totalWord, dev_data_word_len.get(), dev_data_word_len_ps.get(), charPos);
				else 
					MiCuda_ConvertCharToUnsignedInt << < 1024, 128 >> > (dev_data_word.get(), dev_temp_idx_input.data().get(), dev_temp_idx_output.data().get(), totalWord, dev_data_word_len.get(), dev_data_word_len_ps.get(), charPos);
				
				/// sort index by using converted string value
				thrust::sort_by_key(dev_temp_idx_output.begin(), dev_temp_idx_output.end(), dev_temp_idx_input.begin());
			}
			thrust::device_vector<unsigned int> devTemp(totalWord);
			MiCuda_GatherIndex << < 1024, 128 >> > (dev_temp_idx_input.data().get(), dev_data_word_idx.data().get(), devTemp.data().get(), totalWord);
			thrust::copy(devTemp.begin(), devTemp.end(), dev_data_word_idx.begin());
			/*-------------------------------------------*/

			/*----------------- #8 ----------------------*/
			/* hash word and reshuffle based on new index */
			thrust::device_ptr<uint64_t> dev_data_word_hash = thrust::device_malloc<uint64_t>(totalWord);
			thrust::device_ptr<uint64_t> dev_data_word_rehash = thrust::device_malloc<uint64_t>(totalWord);
			cudaMemset(dev_data_word_hash.get(), 0, totalWord * sizeof(uint64_t));
			cudaMemset(dev_data_word_rehash.get(), 0, totalWord * sizeof(uint64_t));
			hash_word << < 1024, 128 >> > (dev_data_word.get(), dev_data_word_hash.get(), dev_data_word_len.get(), dev_data_word_len_ps.get(), totalWord);
			//thrust::device_free(dev_data_word);
			reshuffle << < 1024, 128 >> > (dev_data_word_hash.get(), dev_data_word_rehash.get(), dev_data_word_idx.data().get(), totalWord);
			thrust::device_free(dev_data_word_hash);
			/*----------------- #8 ----------------------*/

			thrust::device_ptr<unsigned int> dev_hash_temp_val = thrust::device_malloc<unsigned int>(totalWord);
			thrust::device_ptr<unsigned int> dev_hash_offset = thrust::device_malloc<unsigned int>(totalWord);
			thrust::device_ptr<uint64_t> dev_data_word_rehash_reduce = thrust::device_malloc<uint64_t>(totalWord);
			thrust::fill_n(dev_hash_temp_val, totalWord, 1);
			cudaMemset(dev_hash_offset.get(), 0, totalWord * sizeof(unsigned int));
			cudaMemset(dev_data_word_rehash_reduce.get(), 0, totalWord * sizeof(uint64_t));
			thrust::equal_to<unsigned int> binary_pred;
			thrust::pair<thrust::device_ptr<uint64_t>, thrust::device_ptr<unsigned int>> new_end;
			new_end = thrust::reduce_by_key(dev_data_word_rehash, dev_data_word_rehash + totalWord, dev_hash_temp_val, dev_data_word_rehash_reduce, dev_hash_offset, binary_pred);

			unsigned int reduceSize = new_end.second - dev_hash_offset;
			thrust::device_ptr<unsigned int> dev_hash_offset_ps = thrust::device_malloc<unsigned int>(reduceSize);
			cudaMemset(dev_hash_offset_ps.get(), 0, reduceSize * sizeof(unsigned int));
			thrust::inclusive_scan(dev_hash_offset, dev_hash_offset + reduceSize, dev_hash_offset_ps);
			// ===========================================================

			
			// =====================	MEMCPY	==========================
			unsigned int* lineIndex = (unsigned int*)malloc(totalWord * sizeof(unsigned int));
			memset(lineIndex, 0, totalWord * sizeof(unsigned int));
			cudaMemcpy(lineIndex, dev_data_word_row_idx.get(), totalWord * sizeof(unsigned int), cudaMemcpyDeviceToHost);

			unsigned int* data_row_len = (unsigned int*)malloc(totalRow * sizeof(unsigned int));
			memset(data_row_len, 0, totalRow * sizeof(unsigned int));
			cudaMemcpy(data_row_len, dev_data_row_len.get(), totalRow * sizeof(unsigned int), cudaMemcpyDeviceToHost);

			unsigned int* data_row_pos = (unsigned int*)malloc(totalRow * sizeof(unsigned int));
			memset(data_row_pos, 0, totalRow * sizeof(unsigned int));
			cudaMemcpy(data_row_pos, dev_data_row_pos.get(), totalRow * sizeof(unsigned int), cudaMemcpyDeviceToHost);

			unsigned int* data_row_pos_idx = (unsigned int*)malloc(totalHaveWord * sizeof(unsigned int));
			memset(data_row_pos_idx, 0, totalHaveWord * sizeof(unsigned int));
			cudaMemcpy(data_row_pos_idx, dev_data_row_pos_idx.get(), totalHaveWord * sizeof(unsigned int), cudaMemcpyDeviceToHost);

			//unsigned int* wordLength = (unsigned int*)malloc(totalWord * sizeof(unsigned int));
			//memset(wordLength, 0, totalWord * sizeof(unsigned int));
			//cudaMemcpy(wordLength, dev_data_word_len.get(), totalWord * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			//
			unsigned int* index = (unsigned int*)malloc(totalWord * sizeof(unsigned int));
			memset(index, 0, totalWord * sizeof(unsigned int));
			cudaMemcpy(index, dev_data_word_idx.data().get(), totalWord * sizeof(unsigned int), cudaMemcpyDeviceToHost);

			//unsigned int* wordLengthPS = (unsigned int*)malloc(totalWord * sizeof(unsigned int));
			//memset(wordLengthPS, 0, totalWord * sizeof(unsigned int));
			//cudaMemcpy(wordLengthPS, dev_data_word_len_ps.get(), totalWord * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			//
			//uint64_t* hashWord = (uint64_t*)malloc(totalWord * sizeof(uint64_t));
			//memset(hashWord, 0, totalWord * sizeof(uint64_t));
			//cudaMemcpy(hashWord, dev_data_word_rehash.get(), totalWord * sizeof(uint64_t), cudaMemcpyDeviceToHost);

			//char* hostWord = (char*)malloc(totalWordSize);
			//memset(hostWord, 0, totalWordSize);
			//cudaMemcpy(hostWord, dev_data_word.get(), totalWordSize, cudaMemcpyDeviceToHost);
			//
			uint64_t* test = (uint64_t*)malloc(reduceSize * sizeof(uint64_t));
			memset(test, 0, reduceSize * sizeof(uint64_t));
			cudaMemcpy(test, dev_data_word_rehash_reduce.get(), reduceSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);

			unsigned int* test1 = (unsigned int*)malloc(reduceSize * sizeof(unsigned int));
			memset(test1, 0, reduceSize * sizeof(unsigned int));
			cudaMemcpy(test1, dev_hash_offset_ps.get(), reduceSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

			unsigned int* test2 = (unsigned int*)malloc(reduceSize * sizeof(unsigned int));
			memset(test2, 0, reduceSize * sizeof(unsigned int));
			cudaMemcpy(test2, dev_hash_offset.get(), reduceSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
			// ===========================================================
			
			unsigned int tempCount = 0;
			unsigned int tempOffset = 0;
			unsigned int* tempIndex;
			printf("Total time: %lf\n", GetSystemTime() - start);
			
			uint64_t sample = 0;
			host_hash_word((void*)"help", sample, 4, 1);
			printf("%llu\n", sample);

			start = GetSystemTime();
			for (int i = 0; i < reduceSize; i++)
			{
				if (test[i] == sample)
				{
					tempCount = test2[i];
					tempIndex = (unsigned int*)malloc(tempCount* sizeof(unsigned int));
					memset(tempIndex, 0, tempCount * sizeof(unsigned int));
					if (i == 0)
						tempOffset = 0;
					else
						tempOffset = test1[i - 1];

					printf("tempOffset : %u\n", tempOffset);
					for (int k = 0; k < tempCount; k++)
					{
						tempIndex[k] = lineIndex[index[tempOffset++]];
					}
				}
			}
			printf("tempCount: %u\n", tempCount);
			printf("Search: %lf\n", GetSystemTime() - start);
			
			// =====================	PRINT	==========================
			FILE* output = fopen("output", "w");/*
			int tick = 0;
			int totalCount = 0;
			for (int i = 0; i < textSize; i++)
			{
				if (lineStat[i])
				{
					//fprintf(output, "%u\n", linePos[tick]);
					fprintf(output, "%u %u %u\n", linePos[tick], wordCount[tick], lineLength[tick]);
					totalCount += wordCount[tick];
					tick++;
				}
			}
			printf("Total word count CPU: %u\n", totalCount);
			printf("%d\n", tick);
			for (int i = 0; i < totalHaveWord; i++)
			{
				fprintf(output, "len: %u count: %u post: %u\n", extLineLength[i], extWordCount[i], extLinePos[i]);
			}
			
			for (int i = 0; i < totalWord; i++)
			{
				fprintf(output, "%u\n", wordLength[i]);
				//if (wordLength[i] > 10)
					//fprintf(output, "i: %d len: %u\n", i, wordLength[i]);
			}
			*//*
			unsigned int idx = 0;
			unsigned int offset = 0;
			char temp[256] = {};
			fprintf(output, "word | index | row |\n", temp, idx, lineIndex[idx]);
			for (int i = 0; i < totalWord; i++)
			{
				idx = index[i];
				if (idx != 0)
				{
					offset = wordLengthPS[idx - 1];
				}

				memset(temp, 0, 256);
				memcpy(temp, hostWord + offset, wordLength[idx]);
				//memcpy(temp, hostWord + offset, wordLength[i]);
				//fprintf(output, "%llu | %u |\n", hashWord[i], index[i]);
				//fprintf(output, "%s | %u | %u |\n", temp, index[i], lineIndex[i]);
				fprintf(output, "%s | %u | %u |\n", temp, idx, lineIndex[idx]);
				offset = 0;
				//offset += wordLength[i];
				//if (wordLength[i] > 10)
				//fprintf(output, "i: %d len: %u\n", i, wordLength[i]);
			}
			*/
			//fprintf(output, "hast | offset_ps | offset |\n");
			//for (int i = 0; i < reduceSize; i++)
			//{
			//	fprintf(output, "%llu | %u | %u |\n", test[i], test1[i], test2[i]);
			//}

			/*for (int i = 0; i < tempCount; i++)
			{
				fprintf(output,"| %u |\n", tempIndex[i]);
			}*/
			char temp[256] = {};
			for (int i = 0; i < tempCount; i++)
			{
				unsigned int idx = data_row_pos_idx[tempIndex[i]];
				unsigned int off = idx == 0 ? 0 : data_row_pos[idx - 1];
				memset(temp, 0, 256);
				memcpy(temp, hostBuffer + ++off, data_row_len[idx]);

				fprintf(output, "| %s | %u |\n", temp, tempIndex[i]);
			}

			fclose(output);
			// ===========================================================
			
			//system("pause");

			// =====================	DELETE	==========================
			remove_buffer(linePos, "host");
			//remove_buffer(hostWord, "host");
			remove_buffer(lineStat, "host");
			remove_buffer(lineCount, "host");
			remove_buffer(hostBuffer, "host");	
			//remove_buffer(wordLength, "host");

			//thrust::device_free(dev_data_word);
			//thrust::device_free(dev_data);
			//thrust::device_free(dev_data_word_end);
			//thrust::device_free(dev_data_word_start);
			thrust::device_free(dev_data_word_len);
			thrust::device_free(dev_data_word_len_ps);
			// ===========================================================
		}

		cudaDeviceReset();
		fclose(input);
	}

	return 0;
}

void remove_buffer(void* ptr, char* property)
{
	if (property == "host")
	{
		if (ptr != NULL)
			free(ptr);
	}
	else if (property == "device")
	{
		if (ptr != NULL)
			cudaFree(ptr);
	}
}

void remove_buffer(thrust::device_ptr<void> ptr, char* property)
{
	if (property == "thrust")
	{
		thrust::device_free(ptr);
	}
}

void compute_block_thread(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);
}

unsigned int compute_sum_kernel(int count, int numThreads, int numBlocks, int maxthreads, int maxblocks, int wkernel, unsigned int *input, unsigned int *output)
{
	unsigned int gpu_result = 0;
	bool needReadBack = true;

	reduce<unsigned int>(count, numThreads, numBlocks, wkernel, input, output);

	int s = numBlocks;
	int kernel = wkernel;

	while (s > 1)
	{
		int threads = 0, blocks = 0;
		compute_block_thread(kernel, s, maxblocks, maxthreads, blocks, threads);

		reduce<unsigned int>(s, threads, blocks, kernel, output, output);

		if (kernel < 3)
		{
			s = (s + threads - 1) / threads;
		}
		else
		{
			s = (s + (threads * 2 - 1)) / (threads * 2);
		}
	}

	if (needReadBack)
	{
		// copy final sum from device to host
		cudaMemcpy(&gpu_result, output, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	}

	return gpu_result;
}
