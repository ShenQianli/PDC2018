__kernel void conv(__global float* input, __global float* filter, __global float* output, int W, int H, int C, int w, int h, int c, int s) {
 
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    int cc, ii, jj, x, y;
    float tmp = 0, tpix = 0, tx, ty, ss;
    for(cc = 0; cc < c; ++cc){
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
    }
    output[i * W * s + j] = tmp;
}

