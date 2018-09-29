#include <iostream>
#include <functional>
#include <caffe2/core/common_omp.h>
#include <caffe2/core/context.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/workspace.h>
#include <caffe2/core/types.h>
#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/graph.h>


#include "bilinear_op.h"
#include <caffe2/core/context_gpu.h>


#include <catch2/catch.hpp>

using namespace std;
using namespace caffe2;

template <class T>
T prod(vector<T>& v) {
    T acc = 1;
    for (t : v) acc *= t;
    return acc;
}

TEST_CASE("Works and stuff", "[basic]") {
    Workspace wrk;
    CPUContext *cctx = new CPUContext();

    NetDef initNet, predNet;

    vector<TIndex> fake_size = {24, 32, 9, 9};


    std::function<DeviceOption*()> allocate_DeviceOption = []() {
        DeviceOption* devop = new DeviceOption();
        devop->set_device_type(CUDA);
        devop->set_cuda_gpu_id(0);
        return devop;
    };

    // Simple init net for testing
    {
        auto op = initNet.add_op();
        op->set_type("GivenTensorFill");
        op->add_output("fa");
        auto val = op->add_arg(), shape = op->add_arg();
        shape->set_name("shape");
        shape->add_ints(fake_size[0]);
        shape->add_ints(fake_size[1]);
        shape->add_ints(fake_size[2]);
        shape->add_ints(fake_size[3]);
        val->set_name("values");
        op->set_allocated_device_option(allocate_DeviceOption());

        int total_eles = prod(fake_size);
        for (int i=0;i<total_eles; i++)
            val->add_floats(i%2);
            //val->add_floats(2);

        op = initNet.add_op();
        op->set_type("GivenTensorFill");
        op->add_output("fb");
        val = op->add_arg(), shape = op->add_arg();
        shape->set_name("shape");
        shape->add_ints(fake_size[0]);
        shape->add_ints(fake_size[1]);
        shape->add_ints(fake_size[2]);
        shape->add_ints(fake_size[3]);
        val->set_name("values");
        op->set_allocated_device_option(allocate_DeviceOption());

        total_eles = prod(fake_size);
        for (int i=0;i<total_eles; i++)
            //val->add_floats(i % fake_size[1]);
            val->add_floats(2);
    }

    // Our operator.
    {
        // Better API from core/graph.h
        auto op = AddOp(&predNet, "BilinearPooling", {"fa", "fb"}, {"pooled_out", "pooled_saved_outer"});
        op->set_allocated_device_option(allocate_DeviceOption());

        auto imgW = op->add_arg(), imgH = op->add_arg();
        imgW->set_name("pooling_scheme");
        imgH->set_s("sum");
    }

    initNet.set_name("test_net_init");
    predNet.set_name("test_net");

    wrk.RunNetOnce(initNet);
    wrk.CreateNet(predNet);

    // Test Running op.
    {
        wrk.RunNet(predNet.name());

        // 
        // Test Outer Product
        //
        auto fa = wrk.GetBlob("fa")->GetMutableTensor(CUDA),
             fb = wrk.GetBlob("fb")->GetMutableTensor(CUDA),
             outer = wrk.GetBlob("pooled_saved_outer")->GetMutableTensor(CUDA);
        int N = fa->dims()[0], C = fa->dims()[1], H = fa->dims()[2], W = fa->dims()[3];
        int HW = H * W;

        // Device memory.
        const float* fa_data = fa->template data<float>();
        const float* fb_data = fb->template data<float>();
        const float* outer_data = outer->template data<float>();
        CUDAContext gctx;

        // Host memory.
        float* tmpouter = new float[outer->size()];
        float* tmpfa = new float[fa->size()];
        float* tmpfb = new float[fb->size()];
        gctx.CopyBytesToCPU(sizeof(float)*outer->size(), outer_data, tmpouter);
        gctx.CopyBytesToCPU(sizeof(float)*fa->size(), fa_data, tmpfa);
        gctx.CopyBytesToCPU(sizeof(float)*fb->size(), fb_data, tmpfb);
        for(int i=0;i<outer->size();i++) {
          int n = i / C / C / HW,
              cc = (i / HW) % (C*C),
              hw = i % HW;
          /*
          std::cout << i << " " << n << " " << cc << " " << hw << ": (" 
              << tmpfa[(n*C+cc/C)*HW+hw] << " "
              << tmpfb[(n*C+cc%C)*HW+hw] << ") --> " << tmpouter[i] << std::endl;
            */

          REQUIRE(
              tmpfa[(n*C+cc/C)*HW+hw] * tmpfb[(n*C+cc%C)*HW+hw] == tmpouter[i]
          );
        }


        //
        // Test Final Pooled Answer
        //

        auto answer = wrk.GetBlob("pooled_out")->GetMutableTensor(CUDA);
        REQUIRE(answer->size() == N*C*C);

        const float* ans_data = answer->template data<float>();
        float* tmpans = new float[answer->size()];
        gctx.CopyBytesToCPU(sizeof(float)*answer->size(), ans_data, tmpans);

        for(int i=0;i<answer->size();i++) {
          int n = i / C / C,
              c1 = (i / C ) % C,
              c2 = i % (C*C) % C;

          //printf("%d: %d %d %d :: %f\n", i, n,c1,c2, tmpans[i]);
        }


        delete[] tmpans;
        delete[] tmpouter;
        delete[] tmpfa;
        delete[] tmpfb;


    }

    std::cout << " DONE. " << std::endl;

}
