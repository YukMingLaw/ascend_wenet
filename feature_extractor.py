import kaldifeat
import _kaldifeat
import torch
import numpy as np

class FeatureExtractor:
    def __init__(self, args):
        self.opts = kaldifeat.FbankOptions()
        self.opts.frame_opts.dither = 0
        self.opts.mel_opts.num_bins = args['num_bins']
        self.opts.frame_opts.frame_shift_ms = args['frame_shift_ms']
        self.opts.frame_opts.samp_freq = args['samp_freq']
        self.feature_extractor = kaldifeat.Fbank(self.opts)

    def get_feature(self, samples, samples_lens, static_sample_shape):
        samples = samples * (1 << 15)
        tensor_sample = torch.tensor(samples, dtype=torch.float32)
        features = self.feature_extractor(tensor_sample)
        expect_feat_len = _kaldifeat.num_frames(samples_lens, self.opts.frame_opts)
        speechs = np.zeros(static_sample_shape, dtype=np.float32) # (1, 2000, 80)
        speechs[:, 0:expect_feat_len, :] = features
        speech_lens = np.array([expect_feat_len], dtype=np.int32)
        return speechs, speech_lens
    
    