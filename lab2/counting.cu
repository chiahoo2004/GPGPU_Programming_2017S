#include <iostream>
#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

using namespace std;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

#define H 10
__device__ int tree[50000000][H];

struct is_char
{
    __host__ __device__
    bool operator()(char x)
    {
        return ( x != '\n' );
    }
};

struct is_not_char
{
    __host__ __device__
    bool operator()(char x)
    {
        return ( x == '\n' );
    }
};

void CountPosition1(const char *text, int *pos, int text_size)
{    
    int *temp;
	cudaMalloc(&temp, sizeof(int)*text_size*2);
	
	thrust::device_ptr<const char> text_d(text);
	thrust::device_ptr<int> pos_d(pos);
    thrust::device_ptr<int> vec_d(temp);
    thrust::device_ptr<int> mask_d(temp+text_size);
    
	cudaMemset((void*)temp, 0, 2*text_size*sizeof(int));
    
    is_char pred_char;
    is_not_char pred_not_char;
    thrust::replace_if(vec_d, vec_d+text_size, text_d, pred_char, 1);
    thrust::replace_if(mask_d, mask_d+text_size, text_d, pred_not_char, 1);
    
    thrust::inclusive_scan(mask_d, mask_d+text_size, mask_d);
    thrust::inclusive_scan_by_key(mask_d, mask_d+text_size, vec_d, pos_d);

}

// Reference: https://github.com/gdoggg2032/GPGPU_Programming_2016S/blob/master/lab1/counting.cu

__global__ void textIdx(int *pos, int text_size)
{
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < text_size) {
            
        int pathIdx = idx;
        int sum = 0;
        int height = 0;
        int mult = 1;
        /*
        if( pathIdx%2==0 && tree[pathIdx][0]>0 && pathIdx!=0 )
            sum++;
        */    
        
        while( tree[pathIdx][height]>0 && height<H-1 && pathIdx>0 ) {
            
            if(pathIdx<=0)
                break;
            
            if( pathIdx%2==0 ) {
                sum += mult;
                pathIdx--;
            }
                
            
            pathIdx = (pathIdx-1)/2;
            height++;
            
            mult *= 2;
            /*
            if(idx==5){
                printf("sum: %d\n",sum);
                printf("pathIdx: %d\n",pathIdx);
                printf("height: %d\n",height);
                printf("mult: %d\n",mult);
            }
            */
        }
        
        
        /*
        while( tree[pathIdx][height]>0 && height>=0 && pathIdx<text_size  ) {
            height--;
            pathIdx = pathIdx*2+2;
        }
        */
        while( height>=0 && pathIdx<text_size && pathIdx>=0 ) {
            
            if( tree[pathIdx][height]>0 ) {//do
                sum += mult;
                pathIdx = pathIdx*2-1;
                /*
                if(idx==5){
                    printf("sum: %d\n",sum);
                    printf("pathIdx: %d\n",pathIdx);
                }
                */
            }
            else {
                pathIdx = pathIdx*2+1;
            }
            height--;
            mult /= 2;
        }
        
        pos[idx] = sum;
            
        
    }
}

__global__ void char2int(const char *text, int text_size)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	if(idx < text_size) {
        if(text[idx] == '\n')
            tree[idx][0] = 0;
        else
            tree[idx][0] = 1;
    }
}

__global__ void buildTree(const char *text, int text_size, int h)
{
	int idx = (blockIdx.x * blockDim.x + threadIdx.x);
	
    if(idx < text_size/pow(2,h))
        tree[idx][h] = tree[idx*2][h-1] && tree[idx*2+1][h-1];
    
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    
    int blocksize = 32; 
    int nblock = CeilDiv(text_size,blocksize);
    
    char2int<<<nblock,blocksize>>>(text, text_size);
    cudaDeviceSynchronize();
    for(int h = 1; h < H; h++)
	{
		buildTree<<<nblock,blocksize>>>(text, text_size, h);
		cudaDeviceSynchronize();
	}
    
    
    textIdx<<<nblock,blocksize>>>(pos, text_size);
    
}
