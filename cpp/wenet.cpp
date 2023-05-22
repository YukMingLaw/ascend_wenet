#include "wenet.h"

Wenet::Wenet(int device_id, const string& encoder_model_path, const string& decoder_model_path, const string& vocabulary_path)
{
    aclError ret = aclInit(nullptr); // Initialize ACL
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to init acl, ret = " << ret <<endl;
    }
    cout << "acl init successfully" << endl;
    ret = aclrtCreateContext(&context_, device_id);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to set current context, ret = " << ret << endl;
    }
    cout << "Create context successfully" << endl;
    ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to set current context, ret = " << ret << endl;
    }
    cout << "set context successfully" << endl;
    ifstream infile;
	infile.open(vocabulary_path, ios::in);
	if (!infile.is_open())
	{
		cout << "read vocab txt fiald" << endl;
	}
	string buf;
	while (getline(infile, buf))
	{
        istringstream str(buf);	
        string out;
        str >> out;
		vocabulary.push_back(out);
	}
    _encoder = make_shared<Encoder>(device_id, encoder_model_path);
    _decoder = make_shared<Decoder>(device_id, decoder_model_path);
}
Wenet::~Wenet()
{
    aclError ret = aclrtDestroyContext(context_);
    if (ret != ACL_ERROR_NONE) {
        cout << "Destroy Context faild, ret = " << ret <<endl;
    }
    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        cout << "Failed to deinit acl, ret = " << ret <<endl;
    }
    cout << "acl deinit successfully" << endl;
}

int Wenet::Compute(std::vector<std::vector<float>>& feature, std::vector<std::string>& results)
{
    EncoderOutData encoder_out_data;
    _encoder->Process(feature, encoder_out_data);
    std::vector<std::vector<double>> log_probs_seq;
    std::vector<std::vector<int>> log_probs_idx;
    int encoder_out_lens = *((int*)(encoder_out_data.encoder_out_lens.data));
    for(int i=0; i < encoder_out_lens; i++)
    {
        int offset = encoder_out_data.beam_log_probs.shape[2];
        float* _start = (float*)(encoder_out_data.beam_log_probs.data) + i * offset;
        float* _end = (float*)(encoder_out_data.beam_log_probs.data) + (i + 1) * offset;
        vector<float> one_beam_log_prob(_start, _end);
        vector<double> temp(one_beam_log_prob.begin(), one_beam_log_prob.end());
        log_probs_seq.push_back(temp);
    }
    for(int i=0; i < encoder_out_lens; i++)
    {
        int offset = encoder_out_data.beam_log_probs_idx.shape[2];
        int* _start = (int*)(encoder_out_data.beam_log_probs_idx.data) + i * offset;
        int* _end = (int*)(encoder_out_data.beam_log_probs_idx.data) + (i + 1) * offset;
        vector<int> one_log_probs_idx(_start, _end);
        log_probs_idx.push_back(one_log_probs_idx);
    }
    PathTrie root;
    int beam_size = encoder_out_data.beam_log_probs.shape[2];
    std::vector<std::pair<double, std::vector<int>>> ctc_beam_serach_result = ctc_beam_search_decoder(log_probs_seq, log_probs_idx, root, true, beam_size, 0);
    std::vector<std::vector<int>> batch_sent;
    for(int i=0; i< ctc_beam_serach_result.size(); i++)
    {
        batch_sent.push_back(ctc_beam_serach_result[i].second);
    }
    results = map_batch(batch_sent, vocabulary, 1);

    DecoderOutData decoder_out_data;
    _decoder->Process(encoder_out_data, ctc_beam_serach_result, vocabulary.size() - 1, decoder_out_data);

    return decoder_out_data.best_index;
}