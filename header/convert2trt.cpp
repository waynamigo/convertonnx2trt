//
// Created by waynamigo on 21-4-23.
//

#include "convert2trt.h"


bool onnxToTRTModel(const std::string& modelpath, // onnx model path
                          unsigned int maxBatchSize,    // max batch size > 4 in DenseDescriptor
                         IHostMemory*& trtModelStream){ // output buffer for TRT file
    // create the builder Ibuilder is from trt deps
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    // create A new Network container for weights and graph from onnnx
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(maxBatchSize);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();


    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());// parser onnx model

    if (!parser->parseFromFile(modelpath, static_cast<int>(gLogger.getReportableSeverity()))){
        // pad onnx to trtnetwork,and translate the tensor of onnx to Trt format
        gLogError << "Failure while parsing ONNX file" << std::endl;
        return false;
    }


    builder->setMaxWorkspaceSize(1_GiB);// set workspace for convert
    config->setMaxWorkspaceSize(1_GiB);

    builder->setFp16Mode(gArgs.runInFp16);// set Inference mode

    samplesCommon::enableDLA(builder, config, gArgs.useDLACore);

    try{
        ICudaEngine* engine = builder->buildCudaEngine(*network); // build cuda engine,param is onnx network
    }catch{
        (engine);
    }

    trtModelStream = engine->serialize();// serialize the engine, then close everything down

    std::ofstream ofs(trtModelName.c_str(), std::ios::out | std::ios::binary);
    ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
    ofs.close();

    engine->destroy();
    network->destroy();
    builder->destroy();
    parser->destroy();
    trtModelStream->destroy();
    cout<<"saved"<<endl
    return true;
}