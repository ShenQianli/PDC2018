#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include<string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

float Random(float start, float end){
    float dis = end - start;
    return start + dis * (rand() / (RAND_MAX + 1.0));
}

float * Random_Array(int L){
    float * array = (float *)malloc(L * sizeof(float));
    for(int i = 0; i < L; ++i)
        array[i] = Random(-1, 1);
    return array;
}

void trans_conv_cl(float * input, int W, int H, int C, float * filter, int w, int h, int c, float * output, int s){
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("trans_conv_kernel_v1.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    //variable
    cl_int ret;
    cl_event event;
    
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1,
                         &device_id, &ret_num_devices);
 
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
 
    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            W * H * C * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            w * h * c * sizeof(float), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            W * s * H * s * sizeof(float), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            W * H * C * sizeof(float), input, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            w * h * c * sizeof(float), filter, 0, NULL, NULL);
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "conv", &ret);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&W);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&H);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&C);
    ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&w);
    ret = clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&h);
    ret = clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&c);
    ret = clSetKernelArg(kernel, 9, sizeof(cl_int), (void *)&s);
    ret = clSetKernelArg(kernel, 10, sizeof(cl_float) * c, NULL);
    
    // Execute the OpenCL kernel on the list
    size_t global_item_size[2] = {H * s, W * s * C};
    size_t local_item_size[2] = {1, c};
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
            global_item_size, local_item_size, 0, NULL, &event);
    
    cl_ulong time_start, time_end;
    double tot_time;
    ret = clFinish(command_queue);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    tot_time = time_end - time_start;
    printf("kernel execution time: %.3f ms\n", tot_time/1000000.0);
    
    // Read the memory buffer on the device to the local variable output
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            W * s * H * s * sizeof(float), output, 0, NULL, NULL);
 
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}

void trans_conv_serial(float * input, int W, int H, int C, float * filter, int w, int h, int c, float * output, int s){
    for(int i = 0; i < H * s; ++i){
        for(int j = 0; j < W * s; ++j){
            float tmp = 0;
            for(int cc = 0; cc < c; ++cc){
                for(int ii = 0; ii < h; ++ii){
                    for(int jj = 0; jj < w; ++jj){
                        float tpix = 0;
                        float ss = 1.0 / s;
                        float tx = (i - (int)(w/2) + ii) * ss;
                        float ty = (j - (int)(h/2) + jj) * ss;
                        if(tx != (int)(tx) || ty != (int)(ty))
                            tpix = 0;
                        else{
                            int x = tx;
                            int y = ty;
                            if(x >= 0 && x < H && y >= 0 && y < W)
                                tpix = input[cc * W * H + x * W + y];
                            else tpix = 0;
                        }
                        tmp += tpix * filter[cc *w * h + ii * w + jj];
                    }
                }
            }
            output[i * (int)(W * s) + j] = tmp;
        }
    }
}

int main(int argc, char *argv[]){
    int W = 200;
    int H = 200;
    int C = 32;
    float * input = Random_Array(W * H * C);
    int w = 5;
    int h = 5;
    int c = 32;
    float * filter = Random_Array(w * h * c);
    int s = 2;
    float * output = (float *)malloc(W * s * H * s * sizeof(float));
    clock_t start, stop;
    
    start = clock();
    trans_conv_cl(input, W, H, C, filter, w, h, c, output, s);
    stop = clock();
    printf("v1 total time: %f ms\n",(stop-start)*1000.0/CLOCKS_PER_SEC);
    start = clock();
}





