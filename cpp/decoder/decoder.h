#pragma once

#include <iostream>
#include <string.h>
#include "acl/acl.h"
#include "common/ModelProcess.h"
#include "encoder/encoder.h"
#include <algorithm>

using namespace std;

typedef struct
{
    long best_index;
}DecoderOutData;

class Decoder{
public:
    Decoder(int deviceId, string modelPath);
    ~Decoder();
    int Process(EncoderOutData& _encoder_out_data, std::vector<std::pair<double, std::vector<int>>>& ctc_beam_serach_result, uint32_t sos_eos_value, DecoderOutData& _decoder_out_data);
private:
    std::vector<std::vector<int>> input_dims;
    std::vector<std::vector<int>> output_dims;
    std::vector<void *> inputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<void *> outputBuffers;
    std::vector<size_t> outputSizes;
    std::shared_ptr<ModelProcess> m_modelProcess;
    aclmdlDesc *m_modelDesc;

    int max_seq_num = 0;
    vector<void*> host_output_data;
    
};