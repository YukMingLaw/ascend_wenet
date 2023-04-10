from swig_decoders import ctc_beam_search_decoder_batch, Scorer, PathTrie, TrieVector, map_batch
import numpy as np
class Scoring:
    def __init__(self, args):
        self.num_processes = args['num_processes'] # multiprocessing.cpu_count()
        self.beam_size = args['beam_size']
        self.cutoff_prob = args['cutoff_prob']
        self.blank_id = args['blank_id']
        _, self.vocab = self.load_vocab(args['vocab_file_path'])
        self.sos = self.eos = len(self.vocab) - 1
        
    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return id2vocab, vocab

    def ctc_calc(self, encoder_out_lens, beam_log_probs, beam_log_probs_idx):
        encoder_out_len = encoder_out_lens[0]
        print('encoder_out_len:', encoder_out_len)
        beam_log_prob = beam_log_probs[0][:encoder_out_len, :].tolist()
        beam_log_prob_idx = beam_log_probs_idx[0][:encoder_out_len, :].tolist()
        root = TrieVector()
        root_dict = {}
        root_dict[0] = PathTrie()
        root.append(root_dict[0])
        batch_start = [True]

        score_hyps = score_hyps = ctc_beam_search_decoder_batch(
                            [beam_log_prob], 
                            [beam_log_prob_idx], 
                            root, 
                            batch_start,
                            self.beam_size,
                            self.num_processes,
                            self.blank_id,
                            45,
                            self.cutoff_prob
                            )
        return score_hyps, self.sos, self.eos

    def rescoring(self, hyps, best_index):
        hyps = np.array(hyps[0][best_index][1], dtype=np.int32).tolist()
        hyps = map_batch([hyps], self.vocab, 1)
        return hyps

        