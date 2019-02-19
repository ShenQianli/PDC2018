#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<string.h>

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
    int W = 1980;
    int H = 1080;
    int C = 32;
    float * input = Random_Array(W * H * C);
    int w = 9;
    int h = 9;
    int c = 32;
    float * filter = Random_Array(w * h * c);
    int s = 2;
    float * output = (float *)malloc(W * s * H * s) * sizeof(float));
    clock_t start, stop;
    
    start = clock();
    trans_conv_serial(input, W, H, C, filter, w, h, c, output, s);
    stop = clock();
    printf("serial total time: %f ms\n",(stop-start)*1000.0/CLOCKS_PER_SEC);
}








