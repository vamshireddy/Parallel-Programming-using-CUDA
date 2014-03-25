// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
__device__ void kernel(float* input, float* output,int len)
{
	__shared__ float XY[2*BLOCK_SIZE];
	
	int t = threadIdx.x;
	
	int start = blockDim.x*blockIdx.x * 2;
	
	if(start + t <len)
	{
		XY[t] = input[start+t];	
		//printf("Copied : XY[i]=%f from input=%f at pos %d\n",XY[t],input[start+t],t);
	}
	else
	{
		XY[t] = 0.0;
		//printf("else :     length = %d Copied : XY[i]=%f from input=%f at pos %d\n",len,XY[t],0,t);
	}
	if(start + blockDim.x + t < len)
	{
		XY[blockDim.x + t] = input[start+ blockDim.x + t];
		//printf("Copied : XY[i]=%f from input=%f at pos %d\n",XY[blockDim.x + t],input[start+t+blockDim.x],blockDim.x + t);
	}
	else
	{
		XY[blockDim.x + t] = 0.0;
		//printf("else :    length : %d Copied : XY[i]=%f from input=%f at pos %d\n",len,XY[blockDim.x + t],0,blockDim.x + t);
	}
	// 1st phase of the reduction starts here
	//printf("The input at %d is %d\n",start+ blockDim.x + t,input[start+ blockDim.x + t]);
	//printf("The input at %d is %d\n",start+ t,input[start+t]);
	for(int stride = 1;stride<= blockDim.x; stride*=2 )
	{
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2 - 1;
		
		if(index < 2*blockDim.x)
		{
			XY[index] += XY[index - stride];
		}
		__syncthreads();
	}
	
	for(int stride = BLOCK_SIZE/2 ; stride > 0 ; stride /=2 )
	{
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2 - 1;
		
		if(index + stride < BLOCK_SIZE )
		{
				XY[index+stride] += XY[index];
		}
	}
	__syncthreads();
	
	output[start+t] = XY[t];
	output[start+ blockDim.x + t] = XY[blockDim.x + t];
	
	//printf("The output at %d is %d\n",start+ blockDim.x + t,output[start+ blockDim.x + t]);
	//printf("The output at %d is %d\n",start + t,output[start+ t]);
}

    
__global__ void scan(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	kernel(input,output,len);
	
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	dim3 block(BLOCK_SIZE,1,1);
	dim3 grid(1,1,1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	
	scan<<<grid,block>>>(deviceInput,deviceOutput,numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


