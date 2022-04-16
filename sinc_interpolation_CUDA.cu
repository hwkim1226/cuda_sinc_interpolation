#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <time.h>
#include <opencv2/opencv.hpp>

#include <bitset>

#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

using namespace std;
using namespace cv;

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

Mat read_BMP_opencv(char* filename, int& w, int& h);

__global__ void sinc2D5K(uchar* buff, uchar* buffer_out, int w, int h, float offset_xx, float offset_yy);
__global__ void sinc2D7K(uchar* buff, uchar* buffer_out, int w, int h, float offset_xx, float offset_yy);


// Subpixel shift (Stream 사용, Sinc function ver.)
/*
// < Parameters >
// f_width, f_height : 전체 이미지의 width와 height
// offset : x 방향과 y 방향으로의 shift 정도
// margin : 제외할 가장자리의 픽셀 수
// img_num : 1 frame을 만드는 crop image의 갯수
// stream_num : CUDA stream의 갯수
// n_iter : 실행 시간 측정을 위한 반복 횟수
// d_data : subpixel shift할 img의 device memory 주소 (d_ : device memory 주소를 의미)
// d_shifted : x방향으로 offset[0], y방향으로 offset[1]만큼 shift한 img
// h_shifted : d_reduced2 값의 host memory 주소(h_ : host memory 주소를 의미)
// result : 결과 img
//
//
// < Functions & Kernels >
// cudaMalloc : Device memory 할당
// cudaMallocHost : Host memory를 pinned memory로 할당
// cudaFree / cudaFreeHost : 할당된 메모리 해제
// cudaMemcpy : 할당된 메모리에 데이터 복사 (host -> device, device -> host)
//
// read_BMP_opencv : .bmp 파일을 opencv Mat으로 리턴
// sinc2D7K : 7*7 sinc interpolation으로 이미지를 subpixel shift하는 CUDA kernel (sinc2D5K는 5*5 kernel)
// ** for문을 사용하면 실행 시간이 더 길게 측정됨
*/


int main()
{
	cudaEvent_t start, stop;
	float  elapsedTime;
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	int f_width, f_height;

	// offset 입력
	float offset[2] = { 0.1f, 0.2f };

	int margin = 2;
	int crop_size = 2048;
	const int stream_num = 32; // ceil(16384.0f / float(crop_size)) * ceil(8192.0f / float(crop_size))
	// 2048 -> 32, 1536 -> 66, 1024 -> 128, 512 -> 512
	int n_iter = 50;

	cout << "crop size = " << crop_size << endl;
	cout << "number of streams = " << stream_num << endl;
	cout << "repeat = " << n_iter << endl;

	cudaStream_t stream[stream_num];

	for (int n = 0; n < stream_num; n++)
	{
		CUDA_SAFE_CALL(cudaStreamCreate(&stream[n]));
	}

	// imread시 메모리를 pinned memory에 할당하도록 설정
	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	Mat* result = new Mat[stream_num];
	Mat* img = new Mat[stream_num];
	uchar* *data = new uchar*[stream_num];
	char buf[256];

	for (int i = 0; i < stream_num; i++)
	{
		sprintf(buf, "test_2048.bmp");
		img[i] = read_BMP_opencv(buf, f_width, f_height);
		data[i] = img[i].data;
	}


	///////////////////////////// 메모리 할당 ////////////////////////////////
	//uchar* *h_data = new uchar*[stream_num];
	uchar* *h_shifted = new uchar*[stream_num];
	uchar* *d_data = new uchar*[stream_num];
	uchar* *d_shifted = new uchar*[stream_num];



	for (int i = 0; i < stream_num; i++)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_data[i], sizeof(uchar) * f_width * f_height));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_shifted[i], sizeof(uchar) * f_width * f_height));

		CUDA_SAFE_CALL(cudaMallocHost((void**)&h_shifted[i], sizeof(uchar) * f_width * f_height));
	}
	/////////////////////////////////////////////////////////////////////////


	dim3 threadsPerBlock(16, 16, 1);
	dim3 numBlocks(int(f_width/threadsPerBlock.x), int(f_height / threadsPerBlock.y), 1);


	CUDA_SAFE_CALL(cudaEventRecord(start, 0));
	///////////////////////////////// Subpixel Shift ///////////////////////////////////////
	// CPU -> GPU, Kernel, GPU -> CPU가 하나의 loop 안에 들어 있어야 함

	for (int i = 0; i < stream_num; i++)
	{
		// CPU -> GPU
		cudaMemcpyAsync(d_data[i], data[i], sizeof(uchar) * f_width * f_height, cudaMemcpyHostToDevice, stream[i]);

		sinc2D7K <<<numBlocks, threadsPerBlock, 0, stream[i]>>> (d_data[i], d_shifted[i], f_width, f_height, offset[0], offset[1]);

		// GPU -> CPU
		cudaMemcpyAsync(h_shifted[i], d_shifted[i], sizeof(uchar) * f_height * f_width, cudaMemcpyDeviceToHost, stream[i]);
	}
	cudaDeviceSynchronize();
	/////////////////////////////////////////////////////////////////////////////////////////

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

	cout << "width : " << f_width << ", height : " << f_height << endl;
	cout << "CUDA stream " << stream_num << "-way result" << endl;
	printf("Average Shift Time: %3.1f ms\n", elapsedTime);

	for (int i = 0; i < stream_num; i++)
	{
		result[i] = Mat(f_height, f_width, CV_8UC1);
		result[i].data = h_shifted[i];
		result[i] = result[i](Range(20, f_height - 20), Range(20, f_width - 20)).clone();

		sprintf(buf, "output_images/test3_shift_%d.bmp", i);
		imwrite(buf, result[i]);
	}

	///////////////////////////////// 메모리 해제 ///////////////////////////////////

	for (int i = 0; i < stream_num; i++)
	{
		CUDA_SAFE_CALL(cudaFree(d_data[i]));
		CUDA_SAFE_CALL(cudaFree(d_shifted[i]));
		CUDA_SAFE_CALL(cudaFreeHost(h_shifted[i]));

		CUDA_SAFE_CALL(cudaStreamDestroy(stream[i]));
	}
	///////////////////////////////////////////////////////////////////////////////
	//waitKey(5000);

	return 0;
}

Mat read_BMP_opencv(char* filename, int& w, int& h)
{
	Mat input_img = imread(filename, 0);
	if (input_img.empty())
		throw "Argument Exception";

	// extract image height and width from header
	int width = input_img.cols;
	int height = input_img.rows;

	//cout << endl;
	//cout << "  Name: " << filename << endl;
	//cout << " Width: " << width << endl;
	//cout << "Height: " << height << endl;

	w = width;
	h = height;

	return input_img;
}

__global__ void sinc2D5K(uchar* buff, uchar* buffer_out, int w, int h, float offset_xx, float offset_yy)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int width = w, height = h;

	float pi = 3.1415926f;
	float offset_x = offset_xx;
	float offset_y = offset_yy;

	if ((x >= 2 && x < width - 2) && (y >= 2 && y < height - 2))
	{
		float val = buff[width*(y - 2) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +

			buff[width*(y - 1) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +

			buff[width*y + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +

			buff[width*(y + 1) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +

			buff[width*(y + 2) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi);

		buffer_out[y * width + x] = (uchar)val;
	}
}

__global__ void sinc2D7K(uchar* buff, uchar* buffer_out, int w, int h, float offset_xx, float offset_yy)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int width = w, height = h;

	float pi = 3.1415926f;
	float offset_x = offset_xx;
	float offset_y = offset_yy;

	if ((x >= 3 && x < width - 3) && (y >= 3 && y < height - 3))
	{
		float val = buff[width*(y - 3) + (x - 3)] * sinf((3.0f - offset_x)*pi) / ((3.0f - offset_x)*pi) * sinf((3.0f - offset_y)*pi) / ((3.0f - offset_y)*pi) +
			buff[width*(y - 3) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((3.0f - offset_y)*pi) / ((3.0f - offset_y)*pi) +
			buff[width*(y - 3) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((3.0f - offset_y)*pi) / ((3.0f - offset_y)*pi) +
			buff[width*(y - 3) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((3.0f - offset_y)*pi) / ((3.0f - offset_y)*pi) +
			buff[width*(y - 3) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((3.0f - offset_y)*pi) / ((3.0f - offset_y)*pi) +
			buff[width*(y - 3) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((3.0f - offset_y)*pi) / ((3.0f - offset_y)*pi) +
			buff[width*(y - 3) + (x + 3)] * sinf((-3.0f - offset_x)*pi) / ((-3.0f - offset_x)*pi) * sinf((3.0f - offset_y)*pi) / ((3.0f - offset_y)*pi) +

			buff[width*(y - 2) + (x - 3)] * sinf((3.0f - offset_x)*pi) / ((3.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +
			buff[width*(y - 2) + (x + 3)] * sinf((-3.0f - offset_x)*pi) / ((-3.0f - offset_x)*pi) * sinf((2.0f - offset_y)*pi) / ((2.0f - offset_y)*pi) +

			buff[width*(y - 1) + (x - 3)] * sinf((3.0f - offset_x)*pi) / ((3.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +
			buff[width*(y - 1) + (x + 3)] * sinf((-3.0f - offset_x)*pi) / ((-3.0f - offset_x)*pi) * sinf((1.0f - offset_y)*pi) / ((1.0f - offset_y)*pi) +

			buff[width*y + (x - 3)] * sinf((3.0f - offset_x)*pi) / ((3.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +
			buff[width*y + (x + 3)] * sinf((-3.0f - offset_x)*pi) / ((-3.0f - offset_x)*pi) * sinf((0.0f - offset_y)*pi) / ((0.0f - offset_y)*pi) +

			buff[width*(y + 1) + (x - 3)] * sinf((3.0f - offset_x)*pi) / ((3.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +
			buff[width*(y + 1) + (x + 3)] * sinf((-3.0f - offset_x)*pi) / ((-3.0f - offset_x)*pi) * sinf((-1.0f - offset_y)*pi) / ((-1.0f - offset_y)*pi) +

			buff[width*(y + 2) + (x - 3)] * sinf((3.0f - offset_x)*pi) / ((3.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +
			buff[width*(y + 2) + (x + 3)] * sinf((-3.0f - offset_x)*pi) / ((-3.0f - offset_x)*pi) * sinf((-2.0f - offset_y)*pi) / ((-2.0f - offset_y)*pi) +

			buff[width*(y + 3) + (x - 3)] * sinf((3.0f - offset_x)*pi) / ((3.0f - offset_x)*pi) * sinf((-3.0f - offset_y)*pi) / ((-3.0f - offset_y)*pi) +
			buff[width*(y + 3) + (x - 2)] * sinf((2.0f - offset_x)*pi) / ((2.0f - offset_x)*pi) * sinf((-3.0f - offset_y)*pi) / ((-3.0f - offset_y)*pi) +
			buff[width*(y + 3) + (x - 1)] * sinf((1.0f - offset_x)*pi) / ((1.0f - offset_x)*pi) * sinf((-3.0f - offset_y)*pi) / ((-3.0f - offset_y)*pi) +
			buff[width*(y + 3) + x] * sinf((0.0f - offset_x)*pi) / ((0.0f - offset_x)*pi) * sinf((-3.0f - offset_y)*pi) / ((-3.0f - offset_y)*pi) +
			buff[width*(y + 3) + (x + 1)] * sinf((-1.0f - offset_x)*pi) / ((-1.0f - offset_x)*pi) * sinf((-3.0f - offset_y)*pi) / ((-3.0f - offset_y)*pi) +
			buff[width*(y + 3) + (x + 2)] * sinf((-2.0f - offset_x)*pi) / ((-2.0f - offset_x)*pi) * sinf((-3.0f - offset_y)*pi) / ((-3.0f - offset_y)*pi) +
			buff[width*(y + 3) + (x + 3)] * sinf((-3.0f - offset_x)*pi) / ((-3.0f - offset_x)*pi) * sinf((-3.0f - offset_y)*pi) / ((-3.0f - offset_y)*pi);

		buffer_out[y * width + x] = (uchar)val;
	}
}