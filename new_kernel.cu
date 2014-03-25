#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5

#define TILE_WIDTH 12
#define BLOCK_WIDTH 16





//@@ INSERT CODE HERE


__global__ void convolution_kernel(float* deviceInputImageData,float* deviceOutputImageData,\
								   float* deviceMaskData,int imageWidth,int imageHeight,int imageChannels,int color)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
	
	int row_o = ty + (blockIdx.y*TILE_WIDTH);
	int col_o = tx + (blockIdx.x*TILE_WIDTH);
	
	int row_i = row_o - 2;
	int col_i = col_o - 2;
	
	if( (row_i>=0) && (col_i>=0) && (row_i<imageHeight) && (col_i<imageWidth))
	{
		Ns[ty][tx] = deviceInputImageData[(row_i*imageWidth+col_i)*3+color];
	}
	else
	{
		Ns[ty][tx] = 0.0f;
	}
	
	__syncthreads();
	
	
	
	float output = 0.0f;
	
	int i,j;
	if( ty< TILE_WIDTH && tx<TILE_WIDTH)
	{
		for(i=0;i<Mask_width;i++)
		{
			for(j=0;j<Mask_width;j++)
			{
				output = output + Ns[ty+i][tx+j]*deviceMaskData[i*Mask_width+j];
				
			}
		}
	}
	__syncthreads();
	
	if(tx < TILE_WIDTH && ty <TILE_WIDTH && row_o < imageHeight && col_o < imageWidth)
	{
		deviceOutputImageData[((row_o*imageWidth)+col_o)*3+color] = output;
	}
	
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	
	dim3 block(BLOCK_WIDTH,BLOCK_WIDTH);
	
	dim3 grid((imageWidth-1)/TILE_WIDTH+1,(imageHeight-1)/TILE_WIDTH+1,3);
	
	convolution_kernel<<<grid,block>>>(deviceInputImageData,deviceOutputImageData,deviceMaskData,imageWidth,imageHeight,imageChannels,0);
	
	cudaThreadSynchronize();
	
	convolution_kernel<<<grid,block>>>(deviceInputImageData,deviceOutputImageData,deviceMaskData,imageWidth,imageHeight,imageChannels,1);
	
	cudaThreadSynchronize();
	
	convolution_kernel<<<grid,block>>>(deviceInputImageData,deviceOutputImageData,deviceMaskData,imageWidth,imageHeight,imageChannels,2);
	
	cudaThreadSynchronize();
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

