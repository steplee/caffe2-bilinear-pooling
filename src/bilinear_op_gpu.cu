#include <caffe2/core/context_gpu.h>
#include "caffe2/operators/operator_fallback_gpu.h"
#include "bilinear_op.h"


__global__ void RunTwoStageOp(
        const float* fa, const float* fb,
        const int N, const int C, const int HW,
        float* out) {

    // TODO logarthimic parallel reduction.

    //const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    CUDA_1D_KERNEL_LOOP(idx, N*HW*C*C) {

        const int n  = idx / (HW) / C / C;
        const int c1  = (idx / (HW) / C) % C;
        const int c2  = (idx / (HW))  % C;
        const int hw = idx % HW;

        float d = fa[ ((n*C)+c2) * HW + hw ] *
                  fb[ ((n*C)+c1) * HW + hw ] /
                  HW;

        // A normal accumulate-write DOES NOT work!
        atomicAdd(out + (n*C*C + c2*C + c1), d);

        // Doesn't work!
        /*out[(n*C + c2)*C + c1] += fa[ ((n*C)+c2) * HW + hw ] *
                                  fb[ ((n*C)+c1) * HW + hw ] /
                                  HW;
        */

        // The fastest way is probably doing each C*C in a block and using shared mem
    }
}

// For each batch, for each spatial location, take of outer product channel-column-vector
// Then pool each spatial location of same batch.
__global__ void RunOuterProduct(
        const float* fa, const float* fb,
        const int N, const int C, const int HW,
        float* out) {

    //const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    CUDA_1D_KERNEL_LOOP(idx, N*HW) {

        // Indexing is a little tricky, since this op takes in NC(HW) but outputs NCC(HW)
        const int n  = idx / (HW);
        const int hw = idx % HW;

        for (int ca=0; ca<C; ca++)
        for (int cb=0; cb<C; cb++) {
            out[(((n*C)+ca)*C+cb)*HW + hw] = fa[((n*C*HW)+hw) + ca*HW] *
                                             fb[((n*C*HW)+hw) + cb*HW];
        }

    }

}

__global__ void RunPool(
        const float* outer_data,
        const int N, const int C, const int HW,
        float* out) {

    CUDA_1D_KERNEL_LOOP(idx, N*C*C) {
        // With a kernel invocation for each N*C*C, sum over each HW

        const int n  = idx / C / C;
        const int cc = idx % (C*C);

        out[idx] = 0;

        for (int hw=0; hw<HW; hw++) {
            out[idx] += outer_data[(n*C*C+cc)*HW + hw] / HW;
        }

    }

}



namespace caffe2 {

    class BilinearPoolingOpCUDA : public Operator<CUDAContext> {
        public:
            USE_OPERATOR_FUNCTIONS(CUDAContext);
            //USE_OPERATOR_CONTEXT_FUNCTIONS;


            BilinearPoolingOpCUDA(const OperatorDef& operator_def, Workspace* ws)
                : Operator<CUDAContext>(operator_def, ws), ws(ws)
        {
        }

        private:

              CUDAContext gctx_;
              Workspace* ws;

              enum PoolingScheme : int {
                SUM = 0,
                MAX = 1
              } poolingScheme_;

        public:

            bool RunOnDevice() override {
                //std::cout << " -- Exec BP on GPU." << std::endl;

                auto fa = Input(0), fb = Input(1);

                int N = fa.dims()[0], C = fa.dims()[1], H = fa.dims()[2], W = fa.dims()[3];
                int HW = H * W;

                const float* fa_data = fa.template data<float>();
                const float* fb_data = fb.template data<float>();

                /*
                std::vector<int> outer_dims = {N,C,C,HW};
                TensorCUDA* outer = Output(1); //(outer_dims, CUDA);
                outer->Resize(outer_dims);
                printf("%ld %ld %ld %ld\n", N,C,H,W);
                std::cout << outer_dims[0]*outer_dims[1]*outer_dims[2]*outer_dims[3] << std::endl;
                std::cout << outer->size() << std::endl;
                float* outer_data = outer->template mutable_data<float>();
                */

                // Final output
                auto out = Output(0);
                std::vector<TIndex> out_dims = {N,C*C};
                out->Resize(out_dims);
                float* out_data = out->template mutable_data<float>();

                cudaMemset(out_data, 0, sizeof(float)*out->size());

                int numThreads = N * HW * C * C;

                RunTwoStageOp<<<
                    CAFFE_GET_BLOCKS(numThreads),
                    CAFFE_CUDA_NUM_THREADS,
                    0,
                    gctx_.cuda_stream()>>>(
                            fa_data,
                            fb_data,
                            N, C, HW,
                            out_data
                            );

                /*
                RunOuterProduct<<<
                    CAFFE_GET_BLOCKS(numThreads),
                    CAFFE_CUDA_NUM_THREADS,
                    0,
                    gctx_.cuda_stream()>>>(
                            fa_data,
                            fb_data,
                            N, C, HW,
                            outer_data
                            );

                numThreads = N * C * C;

                RunPool<<<
                    CAFFE_GET_BLOCKS(numThreads),
                    CAFFE_CUDA_NUM_THREADS,
                    0,
                    gctx_.cuda_stream()>>>(
                            outer_data,
                            N, C, HW,
                            out_data
                            );
                */

                return true;
            }

    };


    REGISTER_CUDA_OPERATOR(
            BilinearPooling,
            BilinearPoolingOpCUDA);

}

