#include <caffe2/core/context_gpu.h>
#include "caffe2/operators/operator_fallback_gpu.h"
#include "bilinear_op.h"


/*
__global__ void RunTwoStageOp(
        const float* fa, const float* fb,
        const int N, const int C, const int HW,
        float* out) {

    // TODO logarthimic parallel reduction.

    //const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    CUDA_1D_KERNEL_LOOP(idx, N*C*C) {

        const int n  = idx / (HW);
        const int hw = idx % HW;

        for (int sa=0; sa<HW; sa++) {
            for (int sb=0; sb<HW; sb++) {
                out[((n*C+sa)*C+sb)*HW + hw] += fa[n*C*HW + hw + sa*HW] *
                                                fb[n*C*HW + hw + sb*HW];

                out[
            }
        }

    }
}
*/

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

                auto fa = Input(0), fb = Input(1);

                int N = fa.dims()[0], C = fa.dims()[1], H = fa.dims()[2], W = fa.dims()[3];
                int HW = H * W;

                const float* fa_data = fa.template data<float>();
                const float* fb_data = fb.template data<float>();

                std::vector<int> outer_dims = {N,C,C,HW};
                TensorCUDA* outer = Output(1); //(outer_dims, CUDA);
                outer->Resize(outer_dims);
                printf("%ld %ld %ld %ld\n", N,C,H,W);
                std::cout << outer_dims[0]*outer_dims[1]*outer_dims[2]*outer_dims[3] << std::endl;
                std::cout << outer->size() << std::endl;
                float* outer_data = outer->template mutable_data<float>();

                // Final output
                auto out = Output(0);
                std::vector<TIndex> out_dims = {N,C,C};
                out->Resize(out_dims);
                float* out_data = out->template mutable_data<float>();

                int numThreads = N * HW;

                /*
                RunTwoStageOp<<<
                    CAFFE_GET_BLOCKS(numThreads),
                    CAFFE_CUDA_NUM_THREADS,
                    0,
                    gctx_.cuda_stream()>>>(
                            fa_data,
                            fb_data,
                            N, C, H, W,
                            out_data
                            );
                */

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

                return true;
            }

    };


    REGISTER_CUDA_OPERATOR(
            BilinearPooling,
            BilinearPoolingOpCUDA);

}

