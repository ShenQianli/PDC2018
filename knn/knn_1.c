#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE 10000

/* Step1: Getting platforms and choose an available one.*/
int getPlatform(cl_platform_id * platform);
/* Step2: Query the platform and get the available devices */
cl_device_id * getCl_device_id(cl_platform_id platform);

int main(void) {
    // read inputs from file
    FILE * fp = fopen("./coordinates.txt", "r");
    // n: number of points
    // dim: dimensions
    // topK: find the topK nearest
    int n, dim, topK, i, j;
    fscanf(fp, "%d\n%d\n%d", &n, &dim, &topK);

    // M: store the coordinates of points, size: n * dim
    // Dist: store the distance of points, size: n * n
    // Res: store the kth nearest points of a point, size: n * topK
    float *M = (float *) malloc(dim * n * sizeof(float));
    float *Dist = (float *)malloc(n * n * sizeof(float));
    int *Res = (int *)malloc(n * topK * sizeof(int));

    // read the points coordinates
    for (i = 0; i < n; ++i){
        for (j = 0; j < dim; ++j){
            fscanf(fp, "%f", M + i * dim + j);
        }
    }
    fclose(fp);
    clock_t start_time = clock();
 
    // Load the kernel source code
    char *source_str;
    size_t source_size;
 
    fp = fopen("knn_kernel_1.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // some variables
    cl_int ret;
    cl_platform_id platform;
    cl_event event;
    cl_ulong time_start, time_end;
    double tot_time1,tot_time2;

    // get a platform
    ret = getPlatform(&platform);
    if (ret == -1)
        exit(1);

    // get available devices
    cl_device_id * devices = getCl_device_id(platform);

    /* check some device infomation, like max work group size. 
    cl_uint max_work_item_dimensions;
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
    printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %u\n", max_work_item_dimensions);
    size_t* max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * max_work_item_dimensions);
    clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL);
    printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: ");
    size_t it = 0;
    for (it = 0; it < max_work_item_dimensions; ++it) 
        printf("%lu\t", max_work_item_sizes[it]); printf("\n");
    free(max_work_item_sizes);
    */    
    // create a context
    cl_context context = clCreateContext(NULL, 1, devices, NULL,NULL,NULL);
    
    // create a command queue (on the first device)
    cl_command_queue command_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &ret);
    
    // create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    
    // Build the program
    ret = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    
    // if program build failed, print log for debugging
    // printf("%d", ret); 
    if (ret == CL_BUILD_PROGRAM_FAILURE){
        size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char * log = (char *) malloc(log_size);
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        exit(1);
    }
    // Create memory buffers on the device
    cl_mem map_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            sizeof(float) * n * dim, NULL, &ret);
    cl_mem dist_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(float) * n * n, NULL, &ret);
    cl_mem res_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(float) * n * topK, NULL, &ret);

    // Copy the coordinates array M to memory buffer
    ret = clEnqueueWriteBuffer(command_queue, map_mem_obj, CL_TRUE, 0,
            sizeof(float) * n * dim, M, 0, NULL, NULL);
    
    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "calc_dist", &ret);
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&map_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dist_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(float) * dim, NULL);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&n);
    ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&dim);
    // execute the kernel
    size_t global_item_size[2] = {n, n * dim};
    // 本来要用三维的{n, n, dim} 但是 workgroup 第三维的上限是128，因此改成二维
    size_t local_item_size[2] = {1, dim};
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_item_size, local_item_size, 0, NULL, &event);
    // calculate kernel time
    ret = clFinish(command_queue);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    tot_time1 = time_end - time_start;

    ret = clEnqueueReadBuffer(command_queue, dist_mem_obj, CL_TRUE, 0,
            sizeof(float) * n * n, Dist, 0, NULL, NULL);
    /*
    // Display the distance to the screen
    for (i = 0 ; i < n; ++i){
        for (j = 0 ; j < i; ++j){
             printf("%d to %d: %.2f\n", i, j, Dist[i * n + j]);
        } 
    }
    */

    // Create the second kernel
    kernel = clCreateKernel(program, "select_k", &ret);
    
    
    // Set the arguments of the second kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dist_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&res_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(int), (void *)&n);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&topK);
    
    // Execute the second kernel
    size_t global_item_size1 = n;
    size_t local_item_size1 = 4;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size1, &local_item_size1, 0, NULL, &event);

    ret = clFinish(command_queue);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    tot_time2 = time_end - time_start;

    // Read the buffer Dist and Res on the device to the local variable Dist
    ret = clEnqueueReadBuffer(command_queue, res_mem_obj, CL_TRUE, 0, 
            sizeof(int) * n * topK, Res, 0, NULL, NULL);
    clock_t stop_time = clock();
    double elapsed = (double)(stop_time - start_time) * 1000.0 / CLOCKS_PER_SEC;
    printf("Kernel1 executing time: %.3fms\n", tot_time1/1000000.0);
    printf("Kernel2 executing time: %.3fms\n", tot_time2/1000000.0);
    printf("Total Time: %.3fms\n", elapsed);

    // Display the result to the screen

    for (i = n - 10; i < n; ++i){
        printf("knn of %d: ", i);
        // PrintPoint(Dist, dim, i, n);
        for (j = 0; j < topK; ++j){
            int v = Res[i * topK + j];
            printf(" %d", v);
            // PrintPoint(Dist, dim, v, n);
        }
        printf("\n");
    }

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(map_mem_obj);
    ret = clReleaseMemObject(dist_mem_obj);
    ret = clReleaseMemObject(res_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(M);
    free(Dist);
    free(Res);
    return 0;
}

int getPlatform(cl_platform_id * platform){
    cl_uint numPlatforms;//the number of platforms
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS){
        printf("error getting platformID\n");
        return -1;
    }

    /* choose the first available platform. */
    if(numPlatforms > 0){
        cl_platform_id * platforms =
            (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform[0] = platforms[0];
        free(platforms);
    }
    else{
        printf("number of platform is 0\n");
        return -1;
    }
}

cl_device_id * getCl_device_id(cl_platform_id platform){
    cl_uint numDevices = 0;
    cl_device_id * devices = NULL;
    // get Device number 
    cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    // if GPU available, get the specific devices
    if (numDevices > 0){
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }
    else printf("no available device.");
    return devices;
}