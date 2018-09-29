from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import (core, dyndep, workspace)

from hypothesis import assume, given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.proto import caffe2_pb2

# Import our ops.
dyndep.InitOpsLibrary('build/libbilinear_op.so')

class TestBilinear(hu.HypothesisTestCase):

    ''' Test GPU '''
    @given(B = st.integers(1, 24),
           C = st.integers(3, 128),
           H = st.integers(7,40),
           W = st.integers(7,40),
           **hu.gcs_gpu_only)
    def test_summarize_columns_gpu(self, B,C,H,W, gc, dc):
        self.run_it(B, C, H,W, gc,dc)

    ''' Test CPU '''
    @given(B = st.integers(1, 12),
           C = st.integers(3, 32),
           H = st.integers(10,40),
           W = st.integers(10,40),
           **hu.gcs_cpu_only)
    def test_bilinear_cpu(self, B,C,H,W, gc, dc):
        self.run_it(B, C, H,W, gc,dc)


    def run_it(self, B,C,H,W, gc,dc):
        def gold_standard(fa,fb):
            #outer = np.zeros([B,C,C,H*W], dtype=np.float32)
            fa = fa.reshape([B,C,H*W])
            fb = fb.reshape([B,C,H*W])
            #xx = xx.transpose(
            '''
            outer = np.zeros([B,H*W,C,C], dtype=np.float32)
            for b in range(B):
                for hw in range(H*W):
                    outer[b,hw] = np.outer(fa[b,:,hw], fb[b,:,hw])
                    #outer[b,hw] = np.ones([C,C])

            pooled = np.zeros([B, C,C])

            for b in range(B):
                for hw in range(H*W):
                    pooled[b] += outer[b,hw] / (H*W)
            '''
            pooled = np.zeros([B,C,C])
            for b in range(B):
                for hw in range(H*W):
                    pooled[b] += np.outer(fa[b,:,hw], fb[b,:,hw]) / (H*W)

            #return pooled.reshape([B,C*C]), outer.transpose(0,2,3,1)
            return pooled.reshape([B,C*C]),



        fa = np.random.randn(B,C,H,W).astype(np.float32)
        fb = np.random.randn(B,C,H,W).astype(np.float32)

        op = core.CreateOperator("BilinearPooling",
                ['fa','fb'],
                #['bp_out', 'bp_saved_outer']
                ['bp_out']
        )

        self.assertReferenceChecks(
                device_option = gc,
                op = op,
                inputs = [fa,fb],
                reference=gold_standard
        )


import unittest
if __name__ == '__main__':
        unittest.main()

