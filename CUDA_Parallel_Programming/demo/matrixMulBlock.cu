#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <vector>
//CUDA RunTime API
#include <cuda_runtime.h>
#include <sys/time.h>
using namespace std;

#define THREAD_NUM 256

#define MATRIX_SIZE 10000


// block数量计算公式
const int blocks_num = MATRIX_SIZE*(MATRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;


// 核函数定义
__global__ static void matMultCUDA(const float* a, const float* b, float* c, int n)
{

    //表示目前的 thread 是第几个 thread（由 0 开始计算）
    const int tid = threadIdx.x;

    //表示目前的 thread 属于第几个 block（由 0 开始计算）
    const int bid = blockIdx.x;

    //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
    const int idx = bid * THREAD_NUM + tid;
    const int row = idx / n;
    const int column = idx % n;

    int i;

    //计算矩阵乘法
    if (row < n && column < n)
    {
        float t = 0;

        for (i = 0; i < n; i++)
        {
            t += a[row * n + i] * b[i * n + column];
        }
        c[row * n + column] = t;
    }
}

#define TILE_WIDTH 20
 
//基于块计算的核函数的具体实现
__global__ void matmul(float *M,float *N,float *P,int width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int Col = bx * TILE_WIDTH + tx;
	int Row = by * TILE_WIDTH + ty;
	
	int Pervalue = 0;
	
	for(int i = 0;i < width / TILE_WIDTH;i++)  //有多少个TILE_WIDTH，每个循环计算一个块的大小
	{
		Mds[ty][tx] = M[Row * width + (i * TILE_WIDTH + tx)];
		Nds[ty][tx] = N[Col + (i * TILE_WIDTH + ty) * width];
		__syncthreads();
		
		
		for(int k = 0;k < TILE_WIDTH;k++) //TILE_WIDTH相乘
			Pervalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}
	
	P[Row * width + Col] = Pervalue;
}

//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{
printf("Device Name : %s.\n", prop.name);
printf("totalGlobalMem : %u.\n", prop.totalGlobalMem);
printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
printf("regsPerBlock : %d.\n", prop.regsPerBlock);
printf("warpSize : %d.\n", prop.warpSize);
printf("memPitch : %d.\n", prop.memPitch);
printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
printf("totalConstMem : %d.\n", prop.totalConstMem);
printf("major.minor : %d.%d.\n", prop.major, prop.minor);
printf("clockRate : %d.\n", prop.clockRate);
printf("textureAlignment : %d.\n", prop.textureAlignment);
printf("deviceOverlap : %d.\n", prop.deviceOverlap);
printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}
//CUDA 初始化
bool InitCUDA()
{
    int count;
    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);
    if (count == 0) 
    {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    printf("There is %d device.\n", count);
    int i;
    for (i = 0; i < count; i++) 
    {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    //打印设备信息
    printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
        {
            if (prop.major >= 1) 
            {
            break;
            }
        }
    }
    if (i == count) 
    {
    fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
    return false;
    }
    cudaSetDevice(i);
    return true;
}
//生成随机矩阵，传入要随机生成的矩阵首地址和维度值。
void matgen(float* a, int n) 
{
    int i, j; 

    for (i = 0; i < n; i++) 
    {
        for (j = 0; j < n; j++) 
        {
            //在产生随机小数时可以使用RAND_MAX。
            a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
        }
    }
}

int main()
{
    cout<<"blocknum:"<<blocks_num<<endl;
    //CUDA 初始化
    if (!InitCUDA()) return 0; 

    //定义矩阵
    float *a, *b, *c, *d;

    int n = MATRIX_SIZE;

    //分配内存
    a = (float*)malloc(sizeof(float)* n * n); 
    b = (float*)malloc(sizeof(float)* n * n); 
    c = (float*)malloc(sizeof(float)* n * n); 
    d = (float*)malloc(sizeof(float)* n * n);

    //设置随机数种子
    srand(0);

    //随机生成矩阵
    matgen(a, n);
    matgen(b, n);

    /*把数据复制到显卡内存中*/
    float *cuda_a, *cuda_b, *cuda_c, *cuda_d;


    //cudaMalloc 取得一块显卡内存 
    cudaMalloc((void**)&cuda_a, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_b, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_c, sizeof(float)* n * n);
    cudaMalloc((void**)&cuda_d, sizeof(float)* n * n);
    //cudaMalloc((void**)&time, sizeof(clock_t)* blocks_num * 2);
    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaMemcpy(cuda_a, a, sizeof(float)* n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(float)* n * n, cudaMemcpyHostToDevice);


    ///使用event计算时间,初始化GPU时间间隔
    float time_elapsed=0;
    ///使用event计算时间，初始化CPU时间间隔
    float time_elapsed_c=0;
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);    //创建Event
    cudaEventCreate(&stop);
    cudaEventRecord( start,0);    //记录当前时间

    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
    matMultCUDA <<< blocks_num, THREAD_NUM, 0 >>>(cuda_a , cuda_b , cuda_c , n);


    cudaEventRecord( stop,0);    //记录当前时间
    cudaEventSynchronize(start);    //Waits for an event to complete.
    cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed,start,stop);    //计算时间差
    cudaEventDestroy(stop);
    cudaEventDestroy(start);    //destory the event
    printf("逐点计算时间%f(ms)\n",time_elapsed);


    /*把结果从显示芯片复制回主内存*/
    //clock_t time_use[blocks_num * 2];
    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(c, cuda_c, sizeof(float)* n * n, cudaMemcpyDeviceToHost);
    //cudaMemcpy(&time_use, time, sizeof(clock_t)* blocks_num * 2, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize()：该方法将停止CPU端线程的执行，直到GPU端完成之前CUDA的任务，包括kernel函数、数据拷贝等。
    // cudaThreadSynchronize()：该方法的作用和cudaDeviceSynchronize()基本相同，但它不是一个被推荐的方法，也许在后期版本的CUDA中会被删除。
    // cudaStreamSynchronize()：这个方法接受一个stream ID，它将阻止CPU执行直到GPU端完成相应stream ID的所有CUDA任务，但其它stream中的CUDA任务可能执行完也可能没有执行完。

    cudaFree(cuda_c);
    const int Nd = 10000;
	int Size = Nd * Nd;
	int width = Nd / 500;
    
	//线程块以及线程的划分
	dim3 gridSize(Nd / width,Nd / width);
	dim3 blockSize(width,width);


    cudaEvent_t start_c,stop_c;
    cudaEventCreate(&start_c);    //创建Event
    cudaEventCreate(&stop_c);
 
    cudaEventRecord( start_c,0);    //记录当前时间
    matmul<<<gridSize,blockSize>>>(cuda_a , cuda_b , cuda_d, Nd); //调用核函数
    cudaEventRecord( stop_c,0);    //记录当前时间

    cudaEventSynchronize(start_c);    //Waits for an event to complete.
    cudaEventSynchronize(stop_c);    //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&time_elapsed_c,start_c,stop_c);    //计算时间差
    cudaEventDestroy(stop_c);
    cudaEventDestroy(start_c);    //destory the event
    printf("分块计算时间%f(ms)\n",time_elapsed_c);
    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(d, cuda_d, sizeof(float)* n * n, cudaMemcpyDeviceToHost);
     cudaFree(cuda_d);


    //验证正确性与精确性

    float max_err = 0;

    float average_err = 0; 


    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (d[i * n + j] != 0)
            { 
                //fabs求浮点数x的绝对值
                float err = fabs((c[i * n + j] - d[i * n + j]) / d[i * n + j]);

                if (max_err < err) max_err = err; 

                average_err += err; 
            } 
        } 
    }

    printf("Max error: %g Average error: %g\n",max_err, average_err / (n * n));
    //cout<<"CPUtime: "<<final_time_c<<endl;

    cout<<"执行时间量级差异： " << float(time_elapsed / time_elapsed_c) <<endl;


return 0;

}