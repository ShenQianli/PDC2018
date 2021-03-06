#define MAXFLOAT 3.4e38

__kernel void calc_dist(__global const float * M, 
                        __global float * Dist,
                        __local float * buf, int n, int dim) {
 
    // each work item calculates the distance in one dimension between point i and j
    
    int i = get_global_id(0);
    int dim_idx = get_local_id(1);
    int j = get_global_id(1)/dim;

    if (i != j && i < n && j < n && dim_idx < dim){
        float val_i = M[dim * i + dim_idx];
        float val_j = M[dim * j + dim_idx];
        buf[dim_idx] = (val_i - val_j) * (val_i - val_j);
        
        // wait all other work item to finish computing
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

        if (dim_idx == 0){
            float dist_sum = 0;
            int k;
            for (k=0; k<dim; ++k){
                dist_sum += buf[k];
            }
            // store the Dist[i][j]
            Dist[i * n + j] = dist_sum;
        }
    }
    else if (i == j){
        if (dim_idx == 0)
            Dist[i * n + i] = 0;
    }
}

__kernel void select_k(__global float * Dist, __global int * Res, int n, int topK ){
    // select the kth smallest of point i;
    int i = get_global_id(0);
    
    // a naive implementation: do k times of find_min

    int iter, idx, Min_idx;
    float Minimum;
    if (i < n){
        for (iter = 0; iter < topK; ++iter){
            Minimum = MAXFLOAT;
            for (idx = 0; idx < n; ++idx){
                if (idx == i)
                    continue;
                float val = Dist[i * n + idx];
                if (val < Minimum){
                    Minimum = val;
                    Min_idx = idx;
                }
                
            }
            Dist[i * n + Min_idx] = MAXFLOAT;
            Res[i * topK + iter] = Min_idx;
        }    
    }
}

typedef struct heapentry{
    float distance;
    int idx;
}heap_entry;

__kernel void select_k_heap(__global float * Dist, __global heap_entry * Res,
                            __local heap_entry * heap, int n, int topK ){

    int i = get_global_id(0), iter;
    int heapSize = 0;
    heap[0].distance = MAXFLOAT;
    for (iter = 0; iter < n; iter++){
        if (iter == i) continue;
        heap_entry element;
        element.idx = iter;
        element.distance = Dist[i * n + iter];
        heapSize++;
        if (heapSize > topK){
            if (element.distance > heap[1].distance){ 
                heapSize-- ;
                continue;
            }
            // else, replace the max element in the heap
            int child, now;

            for (now = 1; now * 2 <= heapSize; now = child) {
                child = now * 2;
                if (child != heapSize && heap[child + 1].distance > heap[child].distance) {
                    child++;
                }
                if (element.distance < heap[child].distance) {
                    heap[now] = heap[child];
                } 
                else
                    break;
            }
            heap[now] = element;
            heapSize-- ;
        }
        else{
            // heap is not full yet, insert the element
            heap[heapSize] = element;
            int now = heapSize;
            while (heap[now / 2].distance < element.distance) {
                heap[now] = heap[now / 2];
                now /= 2;
            }
            heap[now] = element;
        }
    }

    // sort heap
    heap_entry maxElement, lastElement;
    int child, now;
    while(heapSize > 1){
        lastElement = heap[heapSize];
        heap[heapSize--] = heap[1];
        for (now = 1; now * 2 <= heapSize; now = child) {
            child = now * 2;
            if (child != heapSize && heap[child + 1].distance > heap[child].distance) {
                child++;
            }

            if (lastElement.distance < heap[child].distance) {
                heap[now] = heap[child];
            } 
            else
                break;
        }
        heap[now] = lastElement;
    }
    // copy heap to Result
    for (iter = 0; iter < topK; ++iter){
        Res[i * topK + iter] = heap[iter + 1];
    }
}
