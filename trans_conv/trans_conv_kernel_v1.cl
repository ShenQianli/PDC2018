__kernel void conv(__global float* input, __global float* filter, __global float* output, int W, int H, int C, int w, int h, int c, int s, __local float * buf) {
 
    int i = get_global_id(0);
    int j = get_global_id(1) / C;
    int cc = get_local_id(1);
    int ii, jj, x, y;
    float tmp = 0, tpix = 0, tx, ty, ss;
    for(ii = 0; ii < h; ++ii){
        for(jj = 0; jj < w; ++jj){
            tpix = 0;
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
            tmp += tpix * filter[cc * w * h + ii * w + jj];
        }
    }
    buf[cc] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(cc == 0){
        for(ii = 1; ii < c; ++ii){
            buf[0] += buf[ii];
        }
        output[i * W * s + j] = buf[0];
    }
}

