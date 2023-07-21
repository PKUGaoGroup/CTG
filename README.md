# $\mathrm{C_T G}$ program

$\mathrm{C_T G}$ (Hi-C To Geometry) algorithm is a diffusion-based method to obtain reliable geometric information of the chromatin from Hi-C data. Here we provide a program based on CUDA, which can rapidly calculate the $\mathrm{C_T G}$ distance from the Hi-C data.

## Installation and usage

1. This program is based on `CUDA`. Before installation, please make sure that you have installed `CUDA 10.1` or later versions.
2. Download the file `CTG.cu`.
3. Makefile. Run this command in the terminal:

        nvcc CTG.cu -o CTG -lcublas

4. Add the path with `CTG` into your environment variables.
5. Use this command to run the program:

        CTG -i input_matrix_name -o output_matrix_name -alpha alpha -iteration_numbers k
    
    with the meanings of the parameters:
    1. `-i`: The name of the file storing the input matrix. It should be stored in the coordinate format:
            
            bin1    bin2    count

            e.g. 

            2	2	4.78862e+01
            2	3	4.79380e+01
            2	5	3.41081e+01
            2	8	1.97129e+01
            2	9	9.91527e+00
            2	10	2.68997e+00
            2	11	8.69303e+00
            2	12	1.12322e+01
            2	13	5.88250e+00
            2	14	1.05338e+01
        
        **Note that the index of bins should start from 1**.
    2. `-o`: The name of the file to store the output $\mathrm{C_T G}$ matrix. It should be a binary file, and can be read using the following code in python:
  
            import numpy as np
            CTG = np.fromfile(output_matrix_name,dtype='float32')
            N = int(len(CTG)**(1/2))
            CTG = np.reshape(CTG,(N,N))
            
    3. `-alpha`: The parameter in the $\mathrm{C_T G}$ algorithm. It is $\lambda$ in the following equation:

        $$S^{(k)} = \sum_{t=1}^k \mathrm{e}^{-\lambda t} P^t$$
        
        The default value is 0.3.
    4. `-iteration_numbers`: The largest k in the equation above. The default value is 20.
