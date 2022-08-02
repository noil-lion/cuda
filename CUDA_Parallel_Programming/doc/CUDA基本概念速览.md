# CUDA基本概念速览
1. 主流的GPU编程接口
Cuda：Nvidia推出，显卡俗称为N卡。
OpenCL:开源的gpu编程接口，几乎适用于所有的显卡。
DirectCompute：微软开发的gpu编程接口，只能用于Windows系统。

1. 线程模型及硬件对应层次
a. thread
b. block: 含有多个thread，可以为1，2，3 维。
c. grid: 含有多个block，可以1，2维度，一个kernel函数对应一个grid。
d. SP: streaming processor, 最基本的处理单元(硬件概念)。
e. SM : streaming multiprocessor, 多个SP组成一个SM（硬件概念）。
f. 每个thread由每个sp执行，每个block由每个SM执行，一个kernel由一个grid执行。

3. Cuda中的SM，SP，wrap的关系：
SP（streaming Process），SM（streaming multiprocessor）是硬件（GPU hardware）概念。而thread，block，grid，warp是软件上的（Cuda）概念。一块GPU卡中，包含大量的SM，每个SM又包含大量的SP。
硬件：
SM ：Stream Multiprocessor，流多处理器，基本控制指令单元，拥有独立的指令调度电路，一个SM下所有的SP共享同一组控制指令。
SP： Stream Processor，流处理器。基本算数指令单元，没有指令调度电路，只有独立执行的算数电路，包括一个ALU（Arithmetic Logic Unit）和一个FPU（Float Point Unit）。每个SP负责处理固定数量的线程。SP中的线程，共享同一组算数指令，处理不同的数据。这种并行执行的方式，称为SIMD（single intruction multiple data, 单指令多数据）
软件：
warp：线程束，是cuda中的逻辑概念，每个warp对应一个SP，warp size等于SP中可使用的线程个数，通常为32。
一个block对应一个SM, 一个SM可以调度多个block。同一网格（grid）中的所有线程，共享相同的全局内存空间。

4. Cuda编程（线程id索引 + 内存分配）
一个线程需要几个内置的变量（blockIdx，blockDim, threadIdx）来唯一标识，他们都是dim3类型变量。
每个thread有自己私有本地内存（Local Memory）和Register
一个Block内的shared memory，可以被该block内的所有thread共享。
全局内存（Global Memory）：可以被所有Block内的所有thread读写。
常量内存（Constant Memory）和纹理内存（Texture Memory）都属于device只读内存，也是被所有thread共享。常量内存：device只读，host可读可写。
一个Grid可以包含多个Blocks，Blocks的组织方式可以是一维的，二维或者三维的。block包含多个Threads，这些Threads的组织方式也可以是一维，二维或者三维的。CUDA中每一个线程都有一个唯一的标识ID—ThreadIdx，这个ID随着Grid和Block的划分方式的不同而变化。

1. Cuda函数执行环境标识符：host, device, global标识
_ _ host_ _: 程序代码在cpu上执行，与普通的C++无异。
_ _ device_ _: 程序代码在gpu中的线程执行，不能调用普通的cpu函数。
_ _ global_ _: cpu调用，gpu执行，常常以foo<<>>(a)的形式调用，不能运行cpu函数。
不要在.cpp文件中声明_ _ device _ _ 和 _ _ global _ _函数，应该在.cu中声明。
kernel函数返回类型必须为void，必须写在.cu的文件中。

1. Cuda 同步函数（同步指的是cpu与gpu）：
cudaDeviceSynchronize()：该方法将停止CPU端线程的执行，直到GPU端完成之前CUDA的任务，包括kernel函数、数据拷贝等。
cudaStreamSynchronize()：这个方法接受一个stream ID，它将阻止CPU执行，直到GPU端完成相应stream ID的所有CUDA任务，但其它stream中的CUDA任务可能执行完也可能没有执行完。 cudaThreadSynchronize()：该方法的作用和cudaDeviceSynchronize()基本相同，但它不是一个被推荐的方法，也许在后期版本的CUDA中会被删除。
cuda kernel函数是异步执行的，即kernel函数在调用之后会立即把控制权交换给cpu，cpu接着向下执行。

## NVCC编译
kernel可以使用成为PTX的CUDA指令集架构来编写，但用C++来写会更加高效，但无论怎样，想在device上运行还是得用NVCC编译成二进制代码才能在GPU上运行。
nvcc是一个编译器驱动,简化了C++程序或PTX代码代码的编译过程,允许我们使用命令行指令执行不同阶段的程序编译。

### 编译流

1. 离线编译  
NVCC编译的源文件包括host代码和device代码,其编译过程主要有
(将设备代码和host代码分离,将device代码编译成PTX代码或者二进制形式,调用CUDA runtime库函数替换主机代码的<<<...>>>部分语句,然后再调用不同的编译器分别编译。设备端代码由nvcc编译成ptx代码或者二进制代码；主机端代码则将以C文件形式输出，由其他高性能编译器，如ICC、GCC或者其他合适的高性能编译器等进行编译。)

编译获得的exe应用程序就可以链接到已编译的host代码,使用CUDA驱动程序的API来加载和执行PTX代码或cubin对象。使用CUDA驱动API时，可以单独执行ptx代码或者cubin对象，而忽略nvcc编译得到的主机端代码。  
注:  PTX- 即parallel-thread-execution，并行线程执行；是预编译后GPU代码的一种形式.  PTX是独立于gpu架构的，因此可以重用相同的代码适用于不同的GPU架构.

总: 
> nvcc分离代码  
> 将GPU代码部分编译为结构代码形式(PTX)或二进制形式(cubin对象)-可选  
> 修改host代码,使用runtime函数调用代替host代码中的<<<>>>,以加载和启动编译后的GPU代码并启动kernel  
> 修改后的host代码以C++形式输出,让另外的工具编译(GCC什么的).  
2. 即时编译  
作为使用 nvcc 编译 CUDA C++ 设备代码的替代方法，NVRTC 可用于在运行时将 CUDA C++ 设备代码编译为 PTX。 NVRTC 是 CUDA C++ 的运行时编译库；更多信息可以在 NVRTC 用户指南中找到。


### nvcc编译选项
-arch:是--gpu-architecture的缩写,指定CUDA输入文件编译的NVIDIA虚拟GPU架构,可选参数为:'compute_35','compute_37','compute_50', 'compute_52','compute_53','compute_60','compute_61','compute_62','compute_70', 'compute_72','compute_75','compute_80','lto_35','lto_37','lto_50','lto_52', 'lto_53','lto_60','lto_61','lto_62','lto_70','lto_72','lto_75','lto_80', 'sm_35','sm_37','sm_50','sm_52','sm_53','sm_60','sm_61','sm_62','sm_70', 'sm_72','sm_75','sm_80'.

-code: 是选项--gpu-code的缩写,指定NVIDIA GPU的名称,以便对PTX进行组装和优化,

-gencode: --generate-code的缩写,用于指定nvcc在目标代码身材方面的行为.
### 编译生成二进制代码的兼容性
对于不同的体系结构(算力不同的GPU平台),需要指定目标体系结构对应的编译器选项-code生产cubin对象,如-code=sm_35编译生成产生计算力为3.5设备的二进制代码.

编译的二进制代码只能在比当前指定GPU版本算力更高的设备上支持执行


### PTX兼容性
部分PTX指令仅支持高计算力设备
如函数warp shuffle就只能在算力3.0以上的设备上执行  
-arch可以用来指定硬件算力,像上诉函数要指定arch=compute_30

低算力机器编译的PTX代码可以编译为针对高算力的二进制代码.
