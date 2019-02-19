# PDC Project

This is a course project for Introduction to Parallel and Distributed Computation, School of EECS Peking University, 2018 spring

In this project we try to design parallel algorithms for two classical problem, transposed convolution and K-Nearest Neighbors(KNN). We run our programs with GPU and gain obviously better performance than serial ones.

#### Usage

##### Transposed Convolution

Run following commands to compile & run serial program.  

```
$ gcc -o trans_conv_serial trans_conv_serial.c -l OpenCL -std=gnu99
$ ./trans_conv_serial
```

Run following commands to compile & run parallel programs. We provide three different versions of implementation. (x = 0,1,2)  

```
$ gcc -o trans_conv_cl_vx trans_conv_cl_vx.c -l OpenCL -std=gnu99
$ ./trans_conv_cl_vx
```

The serial program outputs total time while parallel ones output running time and total time.

##### KNN

Run `generate_sample.py` to generate data.  

```
$ python generate_sample.py
```
```
$ gcc -o knn_serial.out knn_serial.c
$ ./knn_serial.out
```

Run following commands to compile & run serial program.  

Run following commands to compile & run parallel programs. 

```
$ gcc -o knn_1.out knn_1.c -lOpenCL
$ ./knn_1.out
```
