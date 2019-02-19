__kernel void conv(__global float* input, __global float* filter, __global float* output, int W, int H, int C, int w, int h, int c, int s, __local float * buf) {
    
    int i = get_global_id(0) / c;
    int j = get_global_id(1) / h;
    int cc = get_local_id(0);
    int ii = get_local_id(1);
    int jj = get_local_id(2);
    int x, y, _;
    float tpix = 0, tx, ty, ss;
    
    ss = 1.0 / s;
    tx = (i - (int)(w/2) + ii) * ss;
    ty = (j - (int)(h/2) + jj) * ss;
    if(tx != (int)(tx) || ty != (int)(ty))
        tpix = 0;
    else{
        x = tx;
        y = ty;
        if(x >= 0 && x < H && y >= 0 && y < W)
            tpix = input[cc * W * H + x * W + y];
        else tpix = 0;
    }
    buf[cc * w * h + ii * w + jj] = tpix * filter[cc * w * h + ii * w + jj];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0;
    if(cc * w * h + ii * w + jj == 0){
        for(_ = 0; _ < c * w * h; ++_){
            sum += buf[_];
        }
        output[i * W * s + j] = sum;
    }
}

