
#include "bilinear_op.h"

/*
 * To understand some of this better, see the caffe2 tut, then some operators in caffe2 source repo, then here:
 *    https://github.com/caffe2/caffe2/blob/0dd3284525079f3870df92f61bed3b94eb45ff53/caffe2/core/operator_schema.h
 */

namespace caffe2 {

  OPERATOR_SCHEMA(BilinearPooling)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
        Applies Outer product at each spatial cell of two inputs and pools results.
        )DOC")

    .Arg("pooling_scheme", "How to marginalize spatial dims, must be one of: [sum, max]")

    // Need to find out how to grab blobs from Workspace
    .Input(0, "fa", "Feature extractor A.")
    .Input(1, "fb", "Feature extractor B.")
    .Output(0, "out", "Gathered features of blobs")
    .Output(1, "outer", "Unpooled outer-product needed for gradient");

    REGISTER_CPU_OPERATOR(BilinearPooling, BilinearPoolingOp);
    //REGISTER_CPU_OPERATOR(BilinearPooling, BilinearPoolingOp<CPUContext>);

}
