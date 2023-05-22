#pragma once

#include <iostream>
#include <string.h>
#include "acl/acl.h"
#include "common/ModelProcess.h"

using namespace std;

typedef struct
{
    Tensor encoder_out;
    Tensor encoder_out_lens;
    Tensor ctc_log_probs;
    Tensor beam_log_probs;
    Tensor beam_log_probs_idx;
}EncoderOutData;

class Encoder{
public:
    Encoder(int deviceId, string modelPath);
    ~Encoder();
    int Process(std::vector<std::vector<float>>& feature, EncoderOutData& _encoder_out_data);
private:
    std::vector<std::vector<int>> input_dims;
    std::vector<std::vector<int>> output_dims;
    std::vector<void *> inputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    aclrtContext context_;
    aclrtStream stream_;
    std::shared_ptr<ModelProcess> m_modelProcess;
    aclmdlDesc *m_modelDesc;
    vector<void*> host_output_data;
};
