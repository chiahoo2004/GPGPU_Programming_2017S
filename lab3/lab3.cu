#include "lab3.h"
#include <cstdio>
//#include "Timer.h"


#include <iostream>
using namespace std;


__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

// ref: https://github.com/gdoggg2032/GPGPU_Programming_2016S/blob/master/lab3/lab3.cu

__global__ void CalculateFixed(
		const float *background,
		const float *target,
		const float *mask,
		float *fixed,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
		)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if (yt < ht and xt < wt) {
            
            const int curt = wt*yt+xt;
            const int yb = oy+yt, xb = ox+xt;
            const int curb = wb*yb+xb;
            
            
            
            
            fixed[curt*3+0] = 0;
            fixed[curt*3+1] = 0;
            fixed[curt*3+2] = 0;
            
            if (0 < yt) {
                fixed[curt*3+0] += target[curt*3+0]-target[(curt-wt)*3+0];
                fixed[curt*3+1] += target[curt*3+1]-target[(curt-wt)*3+1];
                fixed[curt*3+2] += target[curt*3+2]-target[(curt-wt)*3+2];
            }
            
            if(yt < ht-1) {
                fixed[curt*3+0] += target[curt*3+0]-target[(curt+wt)*3+0];
                fixed[curt*3+1] += target[curt*3+1]-target[(curt+wt)*3+1];
                fixed[curt*3+2] += target[curt*3+2]-target[(curt+wt)*3+2];
            }
            
            if(0 < xt) {
                fixed[curt*3+0] += target[curt*3+0]-target[(curt-1)*3+0];
                fixed[curt*3+1] += target[curt*3+1]-target[(curt-1)*3+1];
                fixed[curt*3+2] += target[curt*3+2]-target[(curt-1)*3+2];
            }
            
            if(xt < wt-1) {
                fixed[curt*3+0] += target[curt*3+0]-target[(curt+1)*3+0];
                fixed[curt*3+1] += target[curt*3+1]-target[(curt+1)*3+1];
                fixed[curt*3+2] += target[curt*3+2]-target[(curt+1)*3+2];
            }
            
            // 0 < yb and
            // yb < hb-1 and
            // 0 < xb and
            // xb < wb-1 and
            
            // yt == 0 || 
            // yt == ht-1 ||
            // xt == 0 || 
            // xt == wt-1 ||
            
            
            if(yt == 0 || mask[curt-wt] < 127.0f) {
                fixed[curt*3+0] += background[(curb-wb)*3+0];
                fixed[curt*3+1] += background[(curb-wb)*3+1];
                fixed[curt*3+2] += background[(curb-wb)*3+2];
            }
            
            if(yt == ht-1 || mask[curt+wt] < 127.0f) {
                fixed[curt*3+0] += background[(curb+wb)*3+0];
                fixed[curt*3+1] += background[(curb+wb)*3+1];
                fixed[curt*3+2] += background[(curb+wb)*3+2];
            }
            
            if(xt == 0 || mask[curt-1] < 127.0f) {
                fixed[curt*3+0] += background[(curb-1)*3+0];
                fixed[curt*3+1] += background[(curb-1)*3+1];
                fixed[curt*3+2] += background[(curb-1)*3+2];
            }
            
            if(xt == wt-1 || mask[curt+1] < 127.0f) {
                fixed[curt*3+0] += background[(curb+1)*3+0];
                fixed[curt*3+1] += background[(curb+1)*3+1];
                fixed[curt*3+2] += background[(curb+1)*3+2];
            }
            
            
            
            if( mask[curt] < 127.0f ) {
                fixed[curt*3+0] = background[curb*3+0];
                fixed[curt*3+1] = background[curb*3+1];
                fixed[curt*3+2] = background[curb*3+2];
            }
                
                
                
            
                
               
            
        
    }        
}

__global__ void PoissonImageCloningIteration(
        const float *background,
		float *fixed, 
		const float *mask,
		float *buf1, float *buf2, // buf1 -> buf2
		int wt, int ht
		)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if (yt < ht and xt < wt) {
            
            const int curt = wt*yt+xt;
            
            if(mask[curt] > 127.0f) {
            
                buf2[curt*3+0] = 0;
                buf2[curt*3+1] = 0;
                buf2[curt*3+2] = 0;
                
                if (0 < yt and mask[curt-wt] > 127.0f) {
                    buf2[curt*3+0] += buf1[(curt-wt)*3+0];
                    buf2[curt*3+1] += buf1[(curt-wt)*3+1];
                    buf2[curt*3+2] += buf1[(curt-wt)*3+2];
                }
                
                if(yt+1 < ht and mask[curt+wt] > 127.0f) {
                    buf2[curt*3+0] += buf1[(curt+wt)*3+0];
                    buf2[curt*3+1] += buf1[(curt+wt)*3+1];
                    buf2[curt*3+2] += buf1[(curt+wt)*3+2];
                }
                
                if(0 < xt and mask[curt-1] > 127.0f) {
                    buf2[curt*3+0] += buf1[(curt-1)*3+0];
                    buf2[curt*3+1] += buf1[(curt-1)*3+1];
                    buf2[curt*3+2] += buf1[(curt-1)*3+2];
                }
                
                if(xt+1 < wt and mask[curt+1] > 127.0f) {
                    buf2[curt*3+0] += buf1[(curt+1)*3+0];
                    buf2[curt*3+1] += buf1[(curt+1)*3+1];
                    buf2[curt*3+2] += buf1[(curt+1)*3+2];
                }
                
                
                buf2[curt*3+0] += fixed[curt*3+0];
                buf2[curt*3+1] += fixed[curt*3+1];
                buf2[curt*3+2] += fixed[curt*3+2];
                
                
                buf2[curt*3+0] /= 4;
                buf2[curt*3+1] /= 4;
                buf2[curt*3+2] /= 4;
                
            }
            else {
                buf2[curt*3+0] = fixed[curt*3+0];
                buf2[curt*3+1] = fixed[curt*3+1];
                buf2[curt*3+2] = fixed[curt*3+2];
            }
            
            
        
    }
}


__global__ void Downsample(
		const float *original,
		float *sampled,
		const int wt, const int ht,
		int scale
		)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if (yt < ht/scale and xt < wt/scale) {
            
        const int curt = wt*yt+xt;
        
        sampled[curt] = original[curt*scale]; 
         
    }
}

__global__ void Upsample(
		float *sampled,
		float *original,
		const int wt, const int ht,
		int scale
		)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	if (yt < ht and xt < wt) {
            
        const int curt = wt*yt+xt;
        const int curs = wt*yt/scale+xt/scale;
        
        original[curt] = sampled[curs];
            
    }
}


void PoissonImageCloning(
    const float *background,
    const float *target,
    const float *mask,
    float *output,
    const int wb, const int hb, const int wt, const int ht,
    const int oy, const int ox
) {
    
    //Timer timer_count_position;
    //timer_count_position.Start();
    
    // set up
    float *fixed, *buf1, *buf2;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
    
    float *fixed_scaled, *buf1_scaled, *buf2_scaled, *mask_scaled;
    cudaMalloc(&fixed_scaled, 3*wt*ht*sizeof(float)); 
    cudaMalloc(&buf1_scaled, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2_scaled, 3*wt*ht*sizeof(float));
    cudaMalloc(&mask_scaled, wt*ht*sizeof(float)); 
    
    
    // initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    
    /*
    float* fixed = new float[3*wt*ht*sizeof(float)]; 
    float* buf1 = new float[3*wt*ht*sizeof(float)];
    float* buf2 = new float[3*wt*ht*sizeof(float)];
    */
    
    //printf("debug0\n");
    
    CalculateFixed<<<gdim, bdim>>>(
        background, target, mask, fixed,
        wb, hb, wt, ht, oy, ox
    );
    
    cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
    /*
    for(int i=0; i<3*wt*ht; i++)
        buf1[i] = target[i];
    */
    
    //printf("debug1\n");
    
    int level = 8;
    //int iter_num = 5000;
    
    //dim3 gdim(CeilDiv(wt, 32*level), CeilDiv(ht, 16*level)), bdim(32, 16); //??????????????????????????????????????????????????
    
    Downsample<<<gdim, bdim>>>(mask, mask_scaled, wt, ht, level);
    Downsample<<<gdim, bdim>>>(fixed, fixed_scaled, wt, ht, level);
    Downsample<<<gdim, bdim>>>(buf1, buf1_scaled, wt, ht, level);
    
    
    // iterate
    for (int i = 0; i < 235; ++i) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/level, ht/level
        );
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/level, ht/level
        );
    }
    level /= 2;
    //dim3 gdim(CeilDiv(wt, 32*level), CeilDiv(ht, 16*level)), bdim(32, 16);
    Downsample<<<gdim, bdim>>>(mask, mask_scaled, wt, ht, level);
    Downsample<<<gdim, bdim>>>(fixed, fixed_scaled, wt, ht, level);
    Upsample<<<gdim, bdim>>>(buf1, buf1_scaled, wt, ht, 2);
    
    
    for (int i = 0; i < 941; ++i) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/level, ht/level
        );
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/level, ht/level
        );
    }
    level /= 2;
    //dim3 gdim(CeilDiv(wt, 32*level), CeilDiv(ht, 16*level)), bdim(32, 16);
    Downsample<<<gdim, bdim>>>(mask, mask_scaled, wt, ht, level);
    Downsample<<<gdim, bdim>>>(fixed, fixed_scaled, wt, ht, level);
    Upsample<<<gdim, bdim>>>(buf1, buf1_scaled, wt, ht, 2);
    
    
    for (int i = 0; i < 3764; ++i) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/level, ht/level
        );
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/level, ht/level
        );
    }
    Upsample<<<gdim, bdim>>>(buf1, buf1_scaled, wt, ht, 2);
    
    
    
    for (int i = 0; i < 15060; ++i) {
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed, mask, buf1, buf2, wt, ht
        );
        PoissonImageCloningIteration<<<gdim, bdim>>>(
            background, fixed, mask, buf2, buf1, wt, ht
        );
    }
    
    
    
    // copy the image back
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    /*
    for(int i=0; i<3*wb*hb; i++)
        output[i] = background[i];
    */
    
    
    SimpleClone<<<gdim, bdim>>>(
        background, buf1, mask, output,
        wb, hb, wt, ht, oy, ox
    );
    
    /*
    clone(
        background, buf1, mask, output,
        wb, hb, wt, ht, oy, ox
    );
    */
    
    
    // clean up
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
    
    cudaFree(fixed_scaled);
    cudaFree(buf1_scaled);
    cudaFree(buf2_scaled);
    cudaFree(mask_scaled);
    
    //timer_count_position.Pause();
    //printf_timer(timer_count_position);
}
