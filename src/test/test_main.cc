
#include <iostream>
#include <caffe2/core/logging.h>
#include <caffe2/core/init.h>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

int main (int argc, char * argv[]) {
    caffe2::GlobalInit(&argc, &argv);

    return Catch::Session().run( argc, argv );
}
