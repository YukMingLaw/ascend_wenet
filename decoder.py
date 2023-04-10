from aclnet import Net
import numpy as np
class Decoder:
    def __init__(self, args):
        self.beam_size = args['beam_size']
        self.hyps_max_len = args['hyps_max_len']
        self.model = Net(args['device_id'], args['decoder_input_shape'], args['decoder_output_shape'], args['decoder_om_path'])
    
    def forward(self, encoder_out, encoder_out_lens, score_hyps, sos, eos):
        all_hyps = []
        all_ctc_score = []
        max_seq_len = 0
        for seq_cand in score_hyps:
            # if candidates less than beam size
            if len(seq_cand) != self.beam_size:
                seq_cand = list(seq_cand)
                seq_cand += (self.beam_size - len(seq_cand)) * [(-float("INF"), (0,))]
            for score, hyps in seq_cand:
                all_hyps.append(list(hyps))
                all_ctc_score.append(score)
                max_seq_len = max(len(hyps), max_seq_len)

        hyps_max_len = max_seq_len + 2
        if hyps_max_len > self.hyps_max_len:
            print('[ERROR] current hyps_max_len ({0}) ge than 200', hyps_max_len)
            return 0
        decoder_in_ctc_score = np.zeros((1, self.beam_size), dtype=np.float32)
        decoder_in_hyps_pad_sos_eos = np.ones((1, self.beam_size, self.hyps_max_len), dtype=np.int64) * eos
        decoder_in_r_hyps_pad_sos_eos = np.ones((1, self.beam_size, self.hyps_max_len), dtype=np.int64)
        decoder_in_hyps_lens_sos = np.ones((1, self.beam_size), dtype=np.int32)

        for j in range(self.beam_size):
            cur_hyp = all_hyps.pop(0)
            cur_len = len(cur_hyp) + 2
            in_hyp = [sos] + cur_hyp + [eos]
            decoder_in_hyps_pad_sos_eos[0][j][0:cur_len] = in_hyp
            decoder_in_hyps_lens_sos[0][j] = cur_len - 1
            decoder_in_ctc_score[0][j] = all_ctc_score.pop(0)


        result = self.model.run([encoder_out, encoder_out_lens, decoder_in_hyps_pad_sos_eos, decoder_in_hyps_lens_sos, decoder_in_r_hyps_pad_sos_eos, decoder_in_ctc_score])
        return result