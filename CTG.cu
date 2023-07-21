/*
* Copyright 2023 Gao's lab, Peking University, CCME. All rights reserved.
*
* NOTICE TO LICENSEE:
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <string.h>
#include <vector>

__global__ void L1_Distance(const int n,const int i,const float *col_i,const float *matrix,float *dis_col_i)
{
	int j = blockIdx.x;
	if (j < n)
	{
		const float *col_j = &matrix[j*n];

		float temp_record = 0.f;
		for (int k = threadIdx.x; k < n; k = k + blockDim.x)
		{
			temp_record += fabsf(col_i[k] - col_j[k]);
		}
		atomicAdd(&dis_col_i[j], temp_record);
	}
}
int main(int argn, char *argv[])
{
	FILE *matrix_in = NULL;
	FILE *matrix_out = NULL;
	int n = 0;
	int iteration_numbers = 20;
	float alpha = 0.3f;
	cublasHandle_t cuhandle;

	//input parameter
	for (int i = 0; i < argn; i = i + 1)
	{
		if (strcmp(argv[i], "-i") == 0)
		{
			i = i + 1;
			matrix_in = fopen(argv[i], "r");
		}
		else if (strcmp(argv[i], "-iteration_numbers") == 0)
		{
			i = i + 1;
			sscanf(argv[i], "%d", &iteration_numbers);
		}
		else if (strcmp(argv[i], "-alpha") == 0)
		{
			i = i + 1;
			sscanf(argv[i], "%f", &alpha);
		}
		else if (strcmp(argv[i], "-o") == 0)
		{
			i = i + 1;
			matrix_out = fopen(argv[i], "wb");
		}
	}

	//security check
	if (matrix_in == NULL)
	{
		printf("Please input a correct matrix name, after -i\n");
		getchar();
		return 0;
	}
	if (iteration_numbers <= 0)
	{
		printf("Please make sure iteration numbers > 0!\n");
		getchar();
		return 0;
	}

	//read matrix(start from 1)
	std::vector<int> array_i;
	std::vector<int> array_j;
	std::vector<float> array_value;
	int non_zero_count=0;
	while (true)
	{
		int i, j;
		float value;
		int pan = fscanf(matrix_in, "%d %d %f", &i, &j, &value);
		if (pan == EOF)
		{
			break;
		}
		i = i - 1;
		j = j - 1;
		array_i.push_back(i);
		array_j.push_back(j);
		array_value.push_back(value);
		if (i > n)
		{
			n = i;
		}
		if (j > n)
		{
			n = j;
		}
		non_zero_count+=1;
	}
	n = n + 1;
	printf("Matrix size is %d\nNonzero number %d\n", n,non_zero_count);

	//initial GPU and copy data to GPU
	float *h_origin_matrix = NULL;
	h_origin_matrix = (float*)malloc(sizeof(float)*n*n);
	memset(h_origin_matrix,0,sizeof(float)*n*n);
	float *d_origin_matrix = NULL;
	cudaMalloc((void**)&d_origin_matrix, sizeof(float)*n*n);
	for (int k = 0; k < array_i.size(); k = k + 1)
	{
		int i = array_i[k];
		int j = array_j[k];
		float value = array_value[k];
		size_t serial = (size_t)i * n + j;
		h_origin_matrix[serial] = value;
		serial = (size_t)i + n * j;
		h_origin_matrix[serial] = value;
		if (i == j)
		{
			h_origin_matrix[serial] = 0.f;
		}//diagonal element set to 0.f
	}
	cudaError_t cudaerror=cudaMemcpy(d_origin_matrix, h_origin_matrix, sizeof(float)*n*n, cudaMemcpyHostToDevice);
	
	//(matrix normalization)
	std::vector<int>col_zero_record;
	cublasCreate(&cuhandle);
	for (size_t col_i = 0; col_i < n; col_i = col_i + 1)
	{
		float sum;
		cublasSasum(cuhandle, n, &d_origin_matrix[col_i*n], 1, &sum);
		if (fabsf(sum) < 1.e-6f)
		{
			col_zero_record.push_back((int)col_i);
			cudaMemset(&d_origin_matrix[col_i*n], 0, sizeof(float)*n);
		}
		else
		{
			sum = 1.f / sum;
			cublasSscal(cuhandle, n, &sum, &d_origin_matrix[col_i*n], 1);
		}
	}

	//itetration (matrix multiplication)
	const float one = 1.f;
	const float zero = 0.f;
	float *d_sum_matrix = NULL;
	float *d_k_matrix = NULL;
	float *d_k_matrix_copy = NULL;
	cudaMalloc((void**)&d_sum_matrix, sizeof(float)*n*n);
	cudaMemset(d_sum_matrix, 0, sizeof(float)*n*n);
	cudaMalloc((void**)&d_k_matrix, sizeof(float)*n*n);
	cudaMemcpy(d_k_matrix, d_origin_matrix, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&d_k_matrix_copy, sizeof(float)*n*n);
	cudaMemcpy(d_k_matrix_copy, d_origin_matrix, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);
	printf("iteration start!\n");
	for (int iteration_i = 0; iteration_i < iteration_numbers; iteration_i = iteration_i + 1)
	{
		printf("iteration %d!\n",iteration_i);
		float factor = expf(-alpha*(iteration_i + 1));
		cublasSaxpy(cuhandle, n*n, &factor, d_k_matrix, 1, d_sum_matrix, 1);
		cublasSgemm(cuhandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_k_matrix, n, d_origin_matrix, n, &zero, d_k_matrix_copy, n);
		cudaMemcpy(d_k_matrix, d_k_matrix_copy, sizeof(float)*n*n, cudaMemcpyDeviceToDevice);
	}
	printf("iteration done!\n");

	//(calculate L1 distance)
	printf("L1 distance start!\n");
	float *d_col_i = NULL;
	cudaMalloc((void**)&d_col_i, sizeof(float)*n);
	float *h_L1_matrix = NULL;
	h_L1_matrix = (float*)malloc(sizeof(float)*n*n);
	int current_zero_id = 0;
	for (int k = 0; k < n; k = k + 1)
	{
		if (current_zero_id < col_zero_record.size())
		{
			if (col_zero_record[current_zero_id] == k)
			{
				current_zero_id += 1;
				continue;
			}
		}
		cudaMemset(d_col_i, 0, sizeof(float)*n);
		L1_Distance << <n, 256 >> >
			(n, k, &d_sum_matrix[k * n], d_sum_matrix, d_col_i);
		cudaMemcpy(&h_L1_matrix[k * n], d_col_i, sizeof(float)*n, cudaMemcpyDeviceToHost);
		if(k%1000==0)
		{
			printf("L1 distance %d\n",k);
		}
	}
	for (int i = 0; i < col_zero_record.size(); i = i + 1)
	{
		for (int j = 0; j < n; j = j + 1)
		{
			h_L1_matrix[col_zero_record[i] * n + j] = 0.f;
			h_L1_matrix[col_zero_record[i] + n * j] = 0.f;
		}
	}
	printf("L1 distance done!\n");
	

	if (matrix_out == NULL)
	{
		matrix_out = fopen("Distance_Matrix.dat", "wb");
	}
	fwrite(h_L1_matrix, sizeof(float), n*n, matrix_out);
	fclose(matrix_out);

	printf("done!");
	return 0;
}