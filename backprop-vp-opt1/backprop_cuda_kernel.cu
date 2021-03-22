

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"


__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
  float *output_hidden_cuda,
  float *input_hidden_cuda,
  float *hidden_partial_sum,
  int in,
  int hid) 
{
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];


   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;
   
   __syncthreads();

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   __syncthreads();
   
   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads();   
   
   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads();

   }
   
   //__syncthreads();

   input_hidden_cuda[index] = weight_matrix[ty][tx];
   
/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
 
	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}


__global__ void bpnn_adjust_weights_cuda2(float * delta,   
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw)  									
{
  
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   float d = delta[index_x];
   float y = ly[index_y];
   float ww = oldw[index];

   if (ww == 0 && d == 0) {
   } else {
     w[index] += ((ETA * d * y) + (MOMENTUM * ww));
     oldw[index] = ((ETA * d * y) + (MOMENTUM * ww));
   }

   __syncthreads();

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }
}

__global__ void bpnn_adjust_weights_cuda(
										 int hid,         
										 float * ly,      
										 int in,          
										 float * w,       
										 float * oldw)  									
{
  
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;

   float ww = oldw[index];

   if (ww == 0) {
   } else {
     w[index] += (MOMENTUM * ww);
     oldw[index] = (MOMENTUM * ww);
   }

   __syncthreads();

   if (ty == 0 && by ==0){
     w[index_x] += MOMENTUM * oldw[index_x];
     oldw[index_x] = MOMENTUM * oldw[index_x];
   }
}

#endif 
