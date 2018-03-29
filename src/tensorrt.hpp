#pragma once

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>

// TensorRT stuff

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "NvCaffeParser.h"

// CUDA stuff
#include <cuda_runtime.h>




using namespace nvuffparser;
using namespace nvinfer1;

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};





// const int INPUT_H = 299; //inception
// const int INPUT_W = 299; //inception
const int INPUT_H = 1024; //mrt
const int INPUT_W = 2048; //mrt

const int nPixels = INPUT_H*INPUT_W;

// static Logger gLogger;
static std::vector<void*> buffers;

// create tensorrt engine from uff file
nvinfer1::ICudaEngine* createTrtFromUFF(char* modelpath);


// prepare buffer
void prepareBuffer(nvinfer1::ICudaEngine* engine);

// prepare buffer for engine
size_t getBufferSize(Dims d, DataType t);

// execute inference with engine
void inference(nvinfer1::ICudaEngine* engine, int dim_in,  float* data_in, int dim_out, float* data_out);

