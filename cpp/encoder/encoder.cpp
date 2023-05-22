#include "encoder/encoder.h"
#include <sys/time.h>

Encoder::Encoder(int deviceId, string modelPath)
{
    //Load model
    if (m_modelProcess == nullptr) {
        m_modelProcess = std::make_shared<ModelProcess>(deviceId, "");
    }
    aclError ret = m_modelProcess->Init(modelPath);

    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to initialize m_modelProcess, ret = " << ret << endl;
    }
    m_modelDesc = m_modelProcess->GetModelDesc();
    //get model input description and malloc them
    size_t input_num = aclmdlGetNumInputs(m_modelDesc);
    for (size_t i = 0; i < input_num; i++) {
        size_t bufferSize = aclmdlGetInputSizeByIndex(m_modelDesc, i);
        void *inputBuffer = nullptr;
        aclError ret = aclrtMalloc(&inputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
        }
        inputBuffers.push_back(inputBuffer);
        inputSizes.push_back(bufferSize);
        aclmdlIODims dims;
        aclmdlGetInputDims(m_modelDesc, i, &dims);
        std::vector<int> dims_data;
        cout << "input [" << i << "]: [ ";
        for(int j=0; j < dims.dimCount; j++){
            cout << dims.dims[j] << " ";
            dims_data.push_back(dims.dims[j]);
        }
        cout << " ]" << "  bufferSize:"<<bufferSize << endl;
        input_dims.push_back(dims_data);
    }
    //get model output description and malloc them
    size_t output_num = aclmdlGetNumOutputs(m_modelDesc);
    for (size_t i = 0; i < output_num; i++) {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(m_modelDesc, i);
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            cout << "Failed to malloc buffer, ret = " << ret << endl;
        }
        outputBuffers.push_back(outputBuffer);
        outputSizes.push_back(bufferSize);
        aclmdlIODims dims;
        aclmdlGetOutputDims(m_modelDesc, i, &dims);
        std::vector<int> dims_data;
        cout << "output [" << i << "]: [ ";
        for(int j=0; j < dims.dimCount; j++){
            cout << dims.dims[j] << " ";
            dims_data.push_back(dims.dims[j]);
        }
        cout << " ]" <<"  bufferSize:"<<bufferSize << endl;
        output_dims.push_back(dims_data);
    }
    host_output_data = vector<void*>(outputSizes.size());
    for(int i=0; i < outputSizes.size(); i++)
    {
        host_output_data[i] = malloc(outputSizes[i]);
    }
    cout << "finish init AclProcess" << endl;
}

Encoder::~Encoder()
{
    m_modelProcess = nullptr;
    for(int i=0; i < outputSizes.size(); i++)
    {
        free(host_output_data[i]);
    }
    cout << "Destroy Context successfully" << endl;
}

int Encoder::Process(std::vector<std::vector<float>>& feature, EncoderOutData& _encoder_out_data)
{
    aclError ret = ACL_ERROR_NONE;
    if(input_dims.size() !=2 && feature.size() > input_dims[0][1] && output_dims.size() !=5 )
    {
        cout << "check error! input_dims.size() !=2 or output_dims.size !=5 " << " or feature lens[" << feature.size() << "] > model's feature lens limits[" << input_dims[0][1] << "]" <<endl;
    }
    int feature_lens = feature.size();
    size_t host_feature_count = feature_lens * feature[0].size() * sizeof(float);
    uint8_t* host_feature_data = static_cast<uint8_t*>(malloc(host_feature_count));
    for(int i=0; i < feature.size(); i++){
        uint32_t feture_len = feature[i].size() * sizeof(float);
        uint32_t offset = i * feture_len;
        memcpy(static_cast<void*>(host_feature_data + offset), static_cast<void*>(feature[i].data()), feture_len);
    }
    aclrtMemcpy(inputBuffers[0], host_feature_count, host_feature_data, host_feature_count, ACL_MEMCPY_HOST_TO_DEVICE);  
    aclrtMemcpy(inputBuffers[1], inputSizes[1], (void*)&feature_lens, inputSizes[1], ACL_MEMCPY_HOST_TO_DEVICE);
    free(host_feature_data);
    //forward
    // struct timeval start;
    // struct timeval end;
    // gettimeofday(&start,NULL);
    ret = m_modelProcess->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
    if (ret != ACL_ERROR_NONE) {
        cout<<"model run faild.ret = "<< ret <<endl;
        return ret;
    }

    for(int i=0; i < outputSizes.size(); i++)
    {
        aclrtMemcpy(host_output_data[i], outputSizes[i], outputBuffers[i], outputSizes[i], ACL_MEMCPY_DEVICE_TO_HOST);
    }

    _encoder_out_data.encoder_out.shape = output_dims[0];
    _encoder_out_data.encoder_out.data = host_output_data[0];
    _encoder_out_data.encoder_out.size = outputSizes[0];

    _encoder_out_data.encoder_out_lens.shape = output_dims[1];
    _encoder_out_data.encoder_out_lens.data = host_output_data[1];
    _encoder_out_data.encoder_out_lens.size = outputSizes[1];

    _encoder_out_data.ctc_log_probs.shape = output_dims[2];
    _encoder_out_data.ctc_log_probs.data = host_output_data[2];
    _encoder_out_data.ctc_log_probs.size = outputSizes[2];

    _encoder_out_data.beam_log_probs.shape = output_dims[3];
    _encoder_out_data.beam_log_probs.data = host_output_data[3];
    _encoder_out_data.beam_log_probs.size = outputSizes[3];

    _encoder_out_data.beam_log_probs_idx.shape = output_dims[4];
    _encoder_out_data.beam_log_probs_idx.data = host_output_data[4];
    _encoder_out_data.beam_log_probs_idx.size = outputSizes[4];
    return ACL_ERROR_NONE;
}