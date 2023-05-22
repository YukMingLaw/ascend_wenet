#pragma once
#include <iostream>
#include "swig_decoders/ctc_beam_search_decoder.h"
#include "encoder/encoder.h"
#include "decoder/decoder.h"

using namespace std;

class Wenet{
public:
    Wenet(int device_id, const string& encoder_model_path, const string& decoder_model_path, const string& vocabulary_path);
    ~Wenet();
    int Compute(std::vector<std::vector<float>>& feature, std::vector<std::string>& results);
private:
    shared_ptr<Encoder> _encoder;
    shared_ptr<Decoder> _decoder;
    std::vector<string> vocabulary;
    aclrtContext context_;
};