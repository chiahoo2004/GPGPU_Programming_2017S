#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <string.h>
#include <math.h>

#include "lab1.h"
static const int W = 1000;
static const int H = 1000;
static const unsigned NFRAME = 240;

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

#define MAXITERATIONS  (128)
#define MAX_DWELL (MAXITERATIONS)
#define CUT_DWELL (MAX_DWELL / 4)

cuFloatComplex c0;
__device__ cuFloatComplex juliaFunctor(cuFloatComplex p,cuFloatComplex c){
	return cuCaddf(cuCmulf(p,p),c);
}

__device__ int evolveComplexPoint(cuFloatComplex p,cuFloatComplex c){
	int it =1;
	while(it <= MAXITERATIONS && cuCabsf(p) <=4){
		p=juliaFunctor(p,c);
		it++;
	}
	return it;
}

__device__ cuFloatComplex convertToComplex(int x, int y,float zoom,float moveX,float moveY){
	float jx = 1.5 * (x - W / 2) / (0.5 * zoom * W) + moveX;
	float jy = (y - H / 2) / (0.5 * zoom * H) + moveY;
	return make_cuFloatComplex(jx,jy);
}

__global__ void computeJulia(uint8_t* data,uint8_t* dataU,uint8_t* dataV,cuFloatComplex c,float zoom,float moveX,float moveY,int time){
	int i =  blockIdx.x * blockDim.x + threadIdx.x;
	int j =  blockIdx.y * blockDim.y + threadIdx.y;

	if(i<W && j<H){
		cuFloatComplex p = convertToComplex(i,j,zoom,moveX,moveY);
		int dwell = evolveComplexPoint(p,c);
        
        int r,g,b;
        int divide = 12;
        
        if(dwell >= MAX_DWELL) {
            r = 0;
            g = 0;
            b = 0;
        } else {
            // cut at zero
            if(dwell < 0) {
                dwell = 0;
            }
            if(dwell <= MAX_DWELL/divide) {
                r = 255;
                g = time + dwell * 255 / (MAX_DWELL/divide);
                b = 0;
                
            } else if(dwell <= MAX_DWELL*2/divide && dwell > MAX_DWELL/divide) {
                r = 255 - time - (dwell-MAX_DWELL/divide) * 255 / (MAX_DWELL/divide);
                g = 255;
                b = 0;
            } else if(dwell <= MAX_DWELL*3/divide && dwell > MAX_DWELL*2/divide) {
                r = 0;
                g = 255;
                b = time + (dwell-MAX_DWELL*2/divide) * 255 / (MAX_DWELL/divide);
            } else if(dwell <= MAX_DWELL*4/divide && dwell > MAX_DWELL*3/divide) {
                r = 0;
                g = 255 - time - (dwell-MAX_DWELL*3/divide) * 255 / (MAX_DWELL/divide);
                b = 255;
            } else if(dwell <= MAX_DWELL*5/divide && dwell > MAX_DWELL*4/divide) {
                r = time + (dwell-MAX_DWELL*4/divide) * 255 / (MAX_DWELL/divide);
                g = 0;
                b = 255;
            }
            else {
                r = 255;
                g = 0;
                b = 255 - time - (dwell-MAX_DWELL*5/divide) * 255 / (MAX_DWELL/divide);
            }
        }
        
        if(r<0)
            r=0;
        if(r>255)
            r=255;
        if(g<0)
            g=0;
        if(g>255)
            g=255;
        if(b<0)
            b=0;
        if(b>255)
            b=255;
        
        data[i*H+j] = (uint8_t)(0.299*r+0.587*g+0.114*b);
        dataU[i*H+j] = (uint8_t)(-0.169*r-0.331*g+0.5*b+128);
        dataV[i*H+j] = (uint8_t)(0.5*r-0.419*g-0.081*b+128);
        
	}

}

void Lab1VideoGenerator::Generate(uint8_t *yuv) {

    uint8_t *ddata, *ddataU, *ddataV;
	cudaMalloc((void **) &ddata, H*W*sizeof(uint8_t));
	cudaMalloc((void **) &ddataU, H*W*sizeof(uint8_t));
	cudaMalloc((void **) &ddataV, H*W*sizeof(uint8_t));
    
    int blocksizeXY = 32; 
	dim3 blocksize(blocksizeXY, blocksizeXY);
    int nblockXY = W/blocksize.x + (W%blocksize.x ? 1 : 0);
	dim3 nblock( nblockXY , nblockXY );
    
	float incre =0.0000003;//0.00000003
	float inci =-0.0009;//-0.00009
	float startre=-0.591;//-0.75
	float starti=-0.387;//0.09
	float zoom=1.0+0.01*(impl->t);//2.0+0.01*(impl->t)
    float moveX=0.09*log(1+impl->t);//0.09*log(1+impl->t)
    float moveY=0.05*log(1+impl->t);//-0.01*log(1+impl->t)

    c0 = make_cuFloatComplex(startre+(impl->t)*incre,starti+(impl->t)*inci);
    computeJulia<<<nblock,blocksize>>>(ddata,ddataU,ddataV,c0,zoom,moveX,moveY,impl->t);
    
    cudaMemcpy(yuv, ddata, H*W, cudaMemcpyDeviceToDevice); 
	cudaDeviceSynchronize();
    cudaMemcpy(yuv+(H*W), ddataU, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
    cudaMemcpy(yuv+(H*W)+(H*W)/4, ddataV, H*W/4, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
    
    ++(impl->t);
    
}
