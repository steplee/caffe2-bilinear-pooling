#pragma once

#include <cmath>
#include <map>
#include <utility>

#include <caffe2/core/common_omp.h>
#include <caffe2/core/context.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/types.h>
#include <caffe2/operators/gather_op.h>
#include <caffe2/utils/conversions.h>
#include <caffe2/utils/math.h>


namespace caffe2 {

    class BilinearPoolingOp : public Operator<CPUContext> {
        public:
            USE_OPERATOR_FUNCTIONS(CPUContext);

            BilinearPoolingOp(const OperatorDef& operator_def, Workspace* ws)
                : Operator<CPUContext>(operator_def, ws)
                  //img_h(this->template GetSingleArgument<int32_t>("img_h",0))
            {
                CAFFE_ENFORCE(0 and "___only gpu is supported.___");
            }

            ~BilinearPoolingOp() {}


            // Main op code.
            bool RunOnDevice() override {
                auto fa = Input(0), fb = Input(1);

                int N = fa.dims()[0], C = fa.dims()[1], H = fa.dims()[2], W = fa.dims()[3];
                int HW = H * W;

                const float* fa_data = fa.template data<float>();
                const float* fb_data = fb.template data<float>();

                std::vector<int> outer_dims = {N,C,C,HW};
                TensorCPU* outer = Output(1); //(outer_dims, CUDA);
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

                // Outer product.
                for (int n=0; n<N; n++)
                    for (int c1=0; c1<C; c1++)
                        for (int c2=0; c2<C; c2++)
                            for (int hw=0; hw<HW; hw++)
                                outer_data[ ((n*C+c1)*C+c2)*HW + hw ] = fa_data[ (n*C+c1)*HW + hw ] *
                                                                        fb_data[ (n*C+c2)*HW + hw ];

                // Pool.
                for (int n=0; n<N; n++)
                    for (int cc=0; cc<C*C; cc++) {
                        out_data[ (n*C*C) + cc ] = 0;
                        for (int hw=0; hw<HW; hw++)
                            out_data[ (n*C*C) + cc ] += outer_data[ (n*C*C + cc)*HW + hw ];
                    }


                return true;


            }



        private:
            enum PoolingScheme : int {
                SUM = 0,
                MAX = 1
            } poolingScheme;
    };

}
