#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ros/ros.h>
#include "utils.h"

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
    if (severity != Severity::kINFO)
      cout << msg << endl;
  }
} gLogger;

IRuntime *runtime;
ICudaEngine *engine;
IExecutionContext *context;
int inputBindingIndex, outputBindingIndex;
int inputHeight, inputWidth;
bool is_initialized = false;
void *bindings[2];

void setup(string planFilename, string inputName, string outputName) {
  ROS_INFO("setup");
  ifstream planFile(planFilename);
  if(!planFile.is_open()) {
    ROS_INFO("cannot get plan file");
    is_initialized = false;
  } else {
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();

    runtime = createInferRuntime(gLogger);
    engine  = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
    context = engine->createExecutionContext();
    ROS_INFO("load setup finished");

    inputBindingIndex = engine->getBindingIndex(inputName.c_str());
    outputBindingIndex = engine->getBindingIndex(outputName.c_str());
    Dims inputDims, outputDims;
    inputDims = engine->getBindingDimensions(inputBindingIndex);
    outputDims = engine->getBindingDimensions(outputBindingIndex);
    inputHeight = inputDims.d[1];
    inputWidth = inputDims.d[2];

    size_t numInput, numOutput;
    numInput = numTensorElements(inputDims);
    numOutput = numTensorElements(outputDims);

    // host
    float *inputDataHost, *outputDataHost;
    inputDataHost = (float*) malloc(numInput * sizeof(float));
    outputDataHost = (float*) malloc(numOutput * sizeof(float));
    // device
    float *inputDataDevice, *outputDataDevice;
    cudaMalloc(&inputDataDevice, numInput * sizeof(float));
    cudaMalloc(&outputDataDevice, numOutput * sizeof(float));

    is_initialized = true;
    ROS_INFO("initialize finished");
  }
}

void destroy(void) {
  runtime->destroy();
  engine->destroy();
  context->destroy();

  is_initialized = false;
}

void infer(cv::Mat image) {
  ROS_INFO("get");
  /* cvImageToTensor(image, inputDataHost, inputDims); */
}

void test(void) {
  ROS_INFO("inside cu");
  cudaDeviceSynchronize();
}

