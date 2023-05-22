from feature_extractor import FeatureExtractor
from encoder import Encoder
from scoring import Scoring
from decoder import Decoder
import soundfile as sf
import time


args = {}
# feature extractor
args['num_bins'] = 80
args['frame_shift_ms'] = 10
args['samp_freq'] = 16000
args['device_id'] = 0
# encoder
args['encoder_input_shape'] = [[1, 2000, 80], [1]]
args['encoder_output_shape'] = [[1, 499, 512], [1], [1,499,5538], [1,499,10], [1,499,10]]
args['encoder_om_path'] = '/home/yolaw/HZJZ/wenet/models/om/encoder_static.om'
# scoring
args['num_processes'] = 1
args['beam_size'] = 10
args['cutoff_prob'] = 0.9999
args['blank_id'] = 0
args['vocab_file_path'] = 'units.txt'
#decoder
args['feature_size'] = 512
args['hyps_max_len'] = 200
args['decoder_input_shape'] = [[1, 499, 512], [1], [1, 10, 200], [1, 10], [1, 10, 200], [1, 10]]
args['decoder_output_shape'] = [[1]]
args['decoder_om_path'] = '/home/yolaw/HZJZ/wenet/models/om/decoder_static.om'


_feature_extractor = FeatureExtractor(args)
_encoder = Encoder(args)
_scoring = Scoring(args)
_decoder = Decoder(args)

waveform, sample_rate = sf.read("/home/yolaw/HZJZ/test3.wav")
    
speech, speech_len = _feature_extractor.get_feature(waveform, len(waveform), (1, 2000, 80))
encoder_out = _encoder.forward(speech, speech_len)
hyps, sos, eos = _scoring.ctc_calc(encoder_out[1], encoder_out[3], encoder_out[4])
best_index = _decoder.forward(encoder_out[0], encoder_out[1], hyps, sos, eos)
best_index = best_index[0][0]
result = _scoring.rescoring(hyps, best_index)
print(result)
