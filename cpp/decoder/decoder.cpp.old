#include "decoder/decoder.h"
#include <sys/time.h>

Decoder::Decoder(int deviceId, string modelPath)
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

    max_seq_num = input_dims[2][2];

    host_output_data = vector<void*>(outputSizes.size());
    for(int i=0; i < outputSizes.size(); i++)
    {
        host_output_data[i] = malloc(outputSizes[i]);
    }
    cout << "finish init AclProcess" << endl;
}

Decoder::~Decoder()
{
    m_modelProcess = nullptr;
    for(int i=0; i < outputSizes.size(); i++)
    {
        free(host_output_data[i]);
    }
    cout << "Destroy Context successfully" << endl;
}

int Decoder::Process(EncoderOutData& _encoder_out_data, std::vector<std::pair<double, std::vector<int>>>& ctc_beam_serach_result, uint32_t sos_eos_value, DecoderOutData& _decoder_out_data)
{
    aclError ret = ACL_ERROR_NONE;

    aclrtMemcpy(inputBuffers[0], inputSizes[0], _encoder_out_data.encoder_out.data, inputSizes[0], ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(inputBuffers[1], inputSizes[1], _encoder_out_data.encoder_out_lens.data, inputSizes[1], ACL_MEMCPY_HOST_TO_DEVICE);

    int64_t* ptr_hyps_pad_sos_eos = (int64_t*)malloc(inputSizes[2]);
    memset(ptr_hyps_pad_sos_eos, 0, inputSizes[2]);
    for(int i=0; i < ctc_beam_serach_result.size(); i++)
    {
        ptr_hyps_pad_sos_eos[i * max_seq_num + 0] = sos_eos_value;
        for(int k=0; k < ctc_beam_serach_result[i].second.size(); k++)
        {
            ptr_hyps_pad_sos_eos[i * max_seq_num + k + 1] = (int64_t)ctc_beam_serach_result[i].second[k];
        }
        int cur_seq_len = ctc_beam_serach_result[i].second.size();
        ptr_hyps_pad_sos_eos[i * max_seq_num + cur_seq_len + 2] = sos_eos_value;
    }
    

    vector<int> hyps_lens_sos;
    for(int i=0; i<ctc_beam_serach_result.size(); i++)
    {
        hyps_lens_sos.push_back((int)ctc_beam_serach_result[i].second.size() + 1);
    }

    int64_t* ptr_r_hyps_pad_sos_eos = (int64_t*)malloc(inputSizes[4]);
    memset(ptr_r_hyps_pad_sos_eos, 0, inputSizes[4]);
    for(int i=0; i<ctc_beam_serach_result.size(); i++)
    {
        ptr_r_hyps_pad_sos_eos[i * max_seq_num + 0] = sos_eos_value;
        reverse(ctc_beam_serach_result[i].second.begin(), ctc_beam_serach_result[i].second.end());
        for(int k=0; k < ctc_beam_serach_result[i].second.size(); k++)
        {
            ptr_r_hyps_pad_sos_eos[i * max_seq_num + k + 1] = (int64_t)ctc_beam_serach_result[i].second[k];
        }
        int cur_seq_len = ctc_beam_serach_result[i].second.size();
        ptr_r_hyps_pad_sos_eos[i * max_seq_num + cur_seq_len + 2] = sos_eos_value;
    }
    

    vector<float> ctc_score;
    for(int i=0; i<ctc_beam_serach_result.size(); i++)
    {
        ctc_score.push_back((float)ctc_beam_serach_result[i].first);
    }
    
    aclrtMemcpy(inputBuffers[2], inputSizes[2], ptr_hyps_pad_sos_eos, inputSizes[2], ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(inputBuffers[3], inputSizes[3], hyps_lens_sos.data(), inputSizes[3], ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(inputBuffers[4], inputSizes[4], ptr_r_hyps_pad_sos_eos, inputSizes[4], ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(inputBuffers[5], inputSizes[5], ctc_score.data(), inputSizes[5], ACL_MEMCPY_HOST_TO_DEVICE);
    free(ptr_hyps_pad_sos_eos);
    free(ptr_r_hyps_pad_sos_eos);
    //forward
    ret = m_modelProcess->ModelInference(inputBuffers, inputSizes, outputBuffers, outputSizes);
    if (ret != ACL_ERROR_NONE) {
        cout<<"model run faild.ret = "<< ret <<endl;
        return ret;
    }

    for(int i=0; i < outputSizes.size(); i++)
    {
        aclrtMemcpy(host_output_data[i], outputSizes[i], outputBuffers[i], outputSizes[i], ACL_MEMCPY_DEVICE_TO_HOST);
    }

    _decoder_out_data.best_index = *((long*)host_output_data[0]);


    return ACL_ERROR_NONE;
}