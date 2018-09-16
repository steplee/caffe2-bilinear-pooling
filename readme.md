# Bilinear Pooling Operator
Implements Bilinear Pooling, for more details read the paper.

## Implementation Approaches
 1. Do a `dger` for each needed pair of vectors.
    - Bad because cublas has no batched `dger` routine.
 2. Do a `gemm` strided & batched with cublas, pooling afterwards.
    - This is probably the best way if you can figure out the needed strided scheme.
 3. Custom kernel for both outer-product and pooling.
    - This is what I chose to do.
    - **TODO**: Do a proper parallel reduction for the pooling, right now it is parallelized `N*C*C`, but there is a lot to be optimized.

## Usage
 - You need two inputs, both must have the same spatial dimensions AND num channels.
 - I think caffe2 requires putting all data you create in forward pass as an output if you need it during backward pass, so ignore output #2, which is the result of the unpooled outer-product.
