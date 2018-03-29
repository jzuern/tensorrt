#include "tensorrt.hpp"

#include <chrono>
#include <array>
#include <tuple>




size_t getBufferSize(Dims d, DataType t)
{
    size_t size = 1;
    for(size_t i=0; i<d.nbDims; i++) size*= d.d[i];

    switch (t) {
        case DataType::kFLOAT: return size*4;
        case DataType::kHALF: return size*2;
        case DataType::kINT8: return size*1;
    }
    assert(0);
    return 0;
}




ICudaEngine* createTrtFromUFF(char* modelpath)
{
    // define plugin factory in order to use custom layers (== TensorRT plugin)
    nvcaffeparser1::IPluginFactory* pluginFactory;
    auto parser = createUffParser();
    parser->setPluginFactory(&pluginFactory);

    std::cout << "test" << std::endl;

    parser->registerInput("inputPlaceholder", DimsCHW(1, INPUT_H, INPUT_W));// mrt graph
    parser->registerOutput("vars/class");// mrt graph

    // parser->registerInput("input", DimsCHW(3, INPUT_H, INPUT_W));// inception graph
    // parser->registerOutput("InceptionV3/Predictions/Reshape_1"); // inception graph

    Logger gLogger;
    std::cout << "test" << std::endl;

    IBuilder* builder = createInferBuilder(gLogger);
    std::cout << "test" << std::endl;

    INetworkDefinition* network = builder->createNetwork();



    if (!parser->parse(modelpath, *network, nvinfer1::DataType::kFLOAT)) {
        std::cout << "Fail to parse UFF model " << modelpath << std::endl;
        exit(0);
    }

    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 30);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine) {
        std::cout << "Unable to create engine" << std::endl;
        exit(0);
    }

    network->destroy();
    builder->destroy();
    parser->destroy();

    std::cout << "Successfully create TensorRT engine from file " << modelpath << std::endl;

    return engine;
}

void inference(ICudaEngine* engine,
               int dim_in,  float* data_in,
               int dim_out, float* data_out)
{
    if(!engine) {
        std::cerr << "Invaild engine. Please remember to create engine first." << std::endl;
        exit(0);
    }
    IExecutionContext* context = engine->createExecutionContext();

    // We have two bindings: input and output.
    assert(engine->getNbBindings() == 2);
    const int input_index = engine->getBindingIndex("input"); // inception
    const int output_index = engine->getBindingIndex("InceptionV3/Predictions/Reshape_1"); //inception
    // const int input_index = engine->getBindingIndex("inputPlaceholder"); // mrt graph
    // const int output_index = engine->getBindingIndex("vars/class"); // mrt graph

    const int input_size = getBufferSize(engine->getBindingDimensions(input_index), engine->getBindingDataType(input_index));
    const int output_size = getBufferSize(engine->getBindingDimensions(output_index), engine->getBindingDataType(output_index));

    int batch_size = 1;

    // Allocate GPU memory for Input / Output data
    void* buffers[2];
    cudaMalloc(&buffers[input_index], batch_size * input_size);
    cudaMalloc(&buffers[output_index], batch_size * output_size);

    // Use CUDA streams to manage the concurrency of copying and executing
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy Input Data to the GPU
    cudaMemcpyAsync(buffers[input_index], data_in,
                    batch_size * input_size,
                    cudaMemcpyHostToDevice, stream);

    // Launch an instance of the GIE compute kernel
    context->enqueue(batch_size, buffers, stream, nullptr);

    // Copy Output Data to the Host
    cudaMemcpyAsync(data_out, buffers[output_index],
                    batch_size * output_size,
                    cudaMemcpyDeviceToHost, stream);

    // It is possible to have multiple instances of the code above
    // in flight on the GPU in different streams.
    // The host can then sync on a given stream and use the results
    cudaStreamSynchronize(stream);
}





int main(int argc, char **argv) {



  // define model path
  char modelpath[] = "/home/jannik/Desktop/hiwi/tensorrt/mrt_graph.uff";
  // char modelpath[] = "/home/jannik/Desktop/hiwi/tensorrt/inceptionv3.uff";

  // create a tensorrt engine from a uff file
  ICudaEngine* engine = createTrtFromUFF(modelpath);

  int dim_in = 299*299*3;// inception
  int dim_out = 1001; // inception

// int dim_in = 299*299*3;// mrt graph
//   int dim_out = 1001; // mrt graph

  float data_in[dim_in];
  for(size_t i = 0; i < dim_in; i++) data_in[i] = 1.0;

  float data_out[dim_out];
  for(size_t i = 0; i < dim_out; i++) data_out[i] = 0.0;


  // // execute inference
  inference(engine, dim_in, data_in, dim_out, data_out);

  for(size_t i = 0; i < dim_out; i++) std::cerr << data_out[i] << std::endl;


}
