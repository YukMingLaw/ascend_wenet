#include <iostream>
#include "feature_extrator/fbank.h"
#include "feature_extrator/wav.h"
#include "wenet.h"
using namespace std;

int main()
{
    //read wav file
    WavReader wav_reader = WavReader("xxx.wav");
    float time = wav_reader.num_samples() * 1.0 / wav_reader.sample_rate();
    float sample_per_second = wav_reader.num_samples() * 1.0 / time;
    cout << "sample_per_second:" << sample_per_second << endl;
    int fream_len_ms = (int)(25 * sample_per_second / 1000.0);
    int frame_shift_ms  = (int)(10 * sample_per_second / 1000.0);
    //create fbank feature extrator
    Fbank fbank_extrator = Fbank(80, wav_reader.sample_rate(), fream_len_ms, frame_shift_ms);
    vector<float> wav_data(wav_reader.data(), wav_reader.data() + wav_reader.num_samples() * wav_reader.num_channel());
    cout << "num_samples:" << wav_reader.num_samples() << endl;
    cout << "num_channel:" << wav_reader.num_channel() << endl;
    cout << "sample_rate:" << wav_reader.sample_rate() << endl;
    std::vector<std::vector<float>> feature;
    //get fbank feature
    int num_feat = fbank_extrator.Compute(wav_data, &feature);
    //wenet init
    Wenet wenet(0, "encoder.om", "decoder.om", "units.txt");
    //wenet process
    std::vector<std::string> results;
    int best_index = wenet.Compute(feature, results);
    cout << "result:" << results[best_index] << endl;
    //print wenet result
}