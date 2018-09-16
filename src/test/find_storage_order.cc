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
#include <caffe2/utils/eigen_utils.h>

using namespace caffe2;
using namespace std;

template <class T>
T prod(vector<T>& v) {
    T acc = 1;
    for (t : v) acc *= t;
    return acc;
}

int main() {

    CPUContext cctx;

    // Test data from vector
    {
        vector<TIndex> dims = {2,3,4,4};
        vector<float> data(prod(dims));

        for (int bi=0;bi<dims[0];bi++)
            for (int ci=0;ci<dims[1];ci++)
                for (int hi=0;hi<dims[2];hi++)
                    for (int wi=0;wi<dims[3];wi++)
                        data[bi*dims[1]*dims[2]*dims[3] +
                            ci*dims[2]*dims[3] +
                            hi*dims[3] +
                            wi]
                            =
                            ((float)bi * 1000) +
                            ((float)ci * 100)  +
                            ((float)hi * 10)   +
                            ((float)wi * 1)    ;


        TensorCPU ten(dims, data, &cctx);

        cout << " -- From vector:\n";
        for (int i=0;i<prod(dims);i++)
            cout << ten.data<float>()[i] << std::endl;

        /*
        //
        // See what spatial_batch_norm_op.cc does.
        //
        int HW = dims[2]*dims[3], NC = dims[0]*dims[1];
        ConstEigenArrayMap<float> map(ten.data<float>(), HW, NC);

        cout << " -- Columns From Eigen Map:\n";
        for (int i=0;i<NC;i++)
            //cout << map.data()[i] << std::endl;
            cout << "i:" << i << " " << map.col(i) << std::endl;
            */
    }

    {
    }

    return 0;
}
