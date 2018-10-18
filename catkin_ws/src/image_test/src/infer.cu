#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ros/ros.h>
#include <utils.h>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char * msg) override {
    if (severity != Severity::kINFO)
      ROS_INFO("%s", msg);
  }
} gLogger;

IRuntime *runtime;
ICudaEngine *engine;
IExecutionContext *context;
int inputBindingIndex, outputBindingIndex;
int inputHeight, inputWidth;
Dims inputDims, outputDims;
bool is_initialized = false;
void *bindings[2];

// pointers
size_t numInput, numOutput;
float *inputDataHost, *outputDataHost;
float *inputDataDevice, *outputDataDevice;

/* void setup(std::string planFilename, std::string inputName, std::string outputName) { */
void setup() {
  ROS_INFO("setup");
  std::ifstream planFile("/home/ubuntu/tensorrt/resnet_test/resnet_v1_50_finetuned_4class_altered_model.plan");
  if(!planFile.is_open()) {
    ROS_INFO("cannot get plan file");
    is_initialized = false;
  } else {
    std::stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    std::string plan = planBuffer.str();

    runtime = createInferRuntime(gLogger);
    engine  = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
    context = engine->createExecutionContext();
    ROS_INFO("load setup finished");

    inputBindingIndex = engine->getBindingIndex("images");
    outputBindingIndex = engine->getBindingIndex("resnet_v1_50/SpatialSqueeze");
    inputDims = engine->getBindingDimensions(inputBindingIndex);
    outputDims = engine->getBindingDimensions(outputBindingIndex);
    inputHeight = inputDims.d[1];
    inputWidth = inputDims.d[2];
    ROS_INFO("input: %d, %d", inputHeight, inputWidth);

    numInput = numTensorElements(inputDims);
    numOutput = numTensorElements(outputDims);

    // host
    inputDataHost = (float*) malloc(numInput * sizeof(float));
    outputDataHost = (float*) malloc(numOutput * sizeof(float));
    // device
    cudaMalloc(&inputDataDevice, numInput * sizeof(float));
    cudaMalloc(&outputDataDevice, numOutput * sizeof(float));

    is_initialized = true;
    ROS_INFO("initialize finished %d, %d", numInput, numOutput);
  }
}

void destroy(void) {
  if(is_initialized) {
    runtime->destroy();
    engine->destroy();
    context->destroy();
    free(inputDataHost);
    free(outputDataHost);
    cudaFree(inputDataDevice);
    cudaFree(outputDataDevice);
  }
  is_initialized = false;
}

void infer(cv::Mat image, float* out) {
  // cvの画像からcnnを走らせる
  ROS_INFO("get");
  cv::resize(image, image, cv::Size(inputWidth, inputHeight));
  cvImageToTensor(image, inputDataHost, inputDims);
  preprocessVgg(inputDataHost, inputDims);
  bindings[inputBindingIndex] = (void*)inputDataDevice;
  bindings[outputBindingIndex] = (void*)outputDataDevice;

  cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);
  context->execute(1, bindings);
  cudaMemcpy(outputDataHost, outputDataDevice, numOutput * sizeof(float), cudaMemcpyDeviceToHost);
  // output
  ROS_INFO("%f %f %f %f", outputDataHost[0], outputDataHost[1], outputDataHost[2], outputDataHost[3]);
  for (int i = 0; i < 4; i++) {
    out[i] = outputDataHost[i];
  }
}

void test(void) {
  ROS_INFO("inside cu");
  cudaDeviceSynchronize();
}



