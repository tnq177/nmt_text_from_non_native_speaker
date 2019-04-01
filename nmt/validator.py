import os
import time
from os.path import join
from os.path import exists

import numpy
import torch

import nmt.utils as ut
import nmt.all_constants as ac


class Validator(object):
    def __init__(self, config, data_manager):
        super(Validator, self).__init__()
        self.logger = ut.get_logger(config['log_file'])
        self.logger.info('Initializing validator')

        self.data_manager = data_manager
        self.beam_alpha = config['beam_alpha']
        self.checkpoints = config['checkpoints']

        self.get_cpkt_path = lambda checkpoint, score: join(config['save_to'], '{}-{}-{}.path'.format(config['model_name'], checkpoint, score))

        self.bleu_script = './scripts/multi-bleu.perl'
        assert exists(self.bleu_script)

        self.save_to = config['save_to']
        if not exists(self.save_to):
            os.makedirs(self.save_to)

        self.perp_curve = {}
        self.perp_curve_path = {}
        self.best_perps = {}
        self.best_perps_path = {}
        for checkpoint in self.checkpoints:
            self.perp_curve_path[checkpoint] = join(self.save_to, '{}_dev_perps.npy'.format(checkpoint))
            self.best_perps_path[checkpoint] = join(self.save_to, '{}_best_perp_scores.npy'.format(checkpoint))
            self.perp_curve[checkpoint] = numpy.array([], dtype=numpy.float32)
            self.best_perps[checkpoint] = numpy.array([], dtype=numpy.float32)
            if exists(self.perp_curve_path[checkpoint]):
                self.perp_curve[checkpoint] = numpy.load(self.perp_curve_path[checkpoint])
            if exists(self.best_perps_path[checkpoint]):
                self.best_perps[checkpoint] = numpy.load(self.best_perps_path[checkpoint])

    def _ids_to_trans(self, trans_ids):
        words = []
        for idx in trans_ids:
            if idx == ac.EOS_ID:
                break
            words.append(self.data_manager.trg_ivocab[idx])

        return u' '.join(words)

    def get_trans(self, probs, scores, symbols):
        sorted_rows = numpy.argsort(scores)[::-1]
        best_trans = None
        beam_trans = []
        for i, r in enumerate(sorted_rows):
            trans_ids = symbols[r]
            trans_out = self._ids_to_trans(trans_ids)
            beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r], probs[r]))
            if i == 0: # highest prob trans
                best_trans = trans_out

        return best_trans, u'\n'.join(beam_trans)

    def evaluate_perp(self, model):
        model.eval()
        start_time = time.time()
        total_loss = 0.
        total_weight = 0.

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            for checkpoint in self.checkpoints:
                self.logger.info('Validating for checkpoint {}'.format(checkpoint))
                ids_file = self.data_manager.dev_files[checkpoint]['ids']
                for batch_data in self.data_manager.get_batch(ids_file=ids_file):
                    src_toks, trg_toks, targets = batch_data

                    src_toks_cuda = src_toks.to(device)
                    trg_toks_cuda = trg_toks.to(device)
                    targets_cuda  = targets.to(device)

                    # get loss
                    ret = model(src_toks_cuda, trg_toks_cuda, targets_cuda)
                    total_loss += ret['nll_loss'].cpu().detach().numpy()
                    total_weight += (targets != ac.PAD_ID).detach().numpy().sum()

                perp = total_loss / total_weight
                perp = numpy.exp(perp) if perp < 300 else float('inf')
                perp = round(perp, ndigits=3)

                self.perp_curve[checkpoint] = numpy.append(self.perp_curve[checkpoint], perp)
                numpy.save(self.perp_curve_path[checkpoint], self.perp_curve[checkpoint])
                self.logger.info('dev perplexity: {}'.format(perp))

        model.train()
        self.logger.info('Calculating dev perp took: {} minutes'.format(float(time.time() - start_time) / 60.0))

    def _is_valid_to_save(self, checkpoint):
        if len(self.best_perps[checkpoint]) == 0:
            return None, True
        else:
            max_idx = numpy.argmax(self.best_perps[checkpoint])
            max_perp = self.best_perps[checkpoint][max_idx]
            if max_perp < self.perp_curve[checkpoint][-1]:
                return None, False
            else:
                return max_idx, True

    def maybe_save(self, model):
        for checkpoint in self.checkpoints:
            remove_idx, save_please = self._is_valid_to_save(checkpoint)
            perp_score = self.perp_curve[checkpoint][-1]

            if remove_idx is not None:
                max_perp = self.best_perps[checkpoint][remove_idx]
                self.logger.info('Current best perps for {}: {}'.format(checkpoint, ', '.join(map(str, numpy.sort(self.best_perps[checkpoint])[::-1]))))
                self.logger.info('Delete {} & use {} instead'.format(max_perp, perp_score))
                self.best_perps[checkpoint] = numpy.delete(self.best_perps[checkpoint], remove_idx)

                # Delete the right checkpoint
                cpkt_path = self.get_cpkt_path(checkpoint, max_perp)

                if exists(cpkt_path):
                    self.logger.info('Delete {}'.format(cpkt_path))
                    os.remove(cpkt_path)

            if save_please:
                self.logger.info('Save {} to list of best perp scores for {}'.format(perp_score, checkpoint))
                self.best_perps[checkpoint] = numpy.append(self.best_perps[checkpoint], perp_score)
                cpkt_path = self.get_cpkt_path(checkpoint, perp_score)
                torch.save(model.state_dict(), cpkt_path)
                self.logger.info('Save new best model to {}'.format(cpkt_path))
                self.logger.info('Best perp scores so far: {}'.format(', '.join(map(str, numpy.sort(self.best_perps[checkpoint])))))

            numpy.save(self.best_perps_path[checkpoint], self.best_perps[checkpoint])

    def validate_and_save(self, model):
        self.logger.info('Start validation')
        self.evaluate_perp(model)
        self.maybe_save(model)

    def translate(self, model, input_file):
        model.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Very redundant
        best_trans_file = input_file + '.best_trans'
        beam_trans_file = input_file + '.beam_trans'

        num_sents = 0
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    num_sents += 1
        all_best_trans = [''] * num_sents
        all_beam_trans = [''] * num_sents
        with torch.no_grad():
            self.logger.info('Start translating {}'.format(input_file))
            start = time.time()
            count = 0
            for (src_toks, original_idxs) in self.data_manager.get_trans_input(input_file):
                src_toks_cuda = src_toks.to(device)
                rets = model.beam_decode(src_toks_cuda)

                for i, ret in enumerate(rets):
                    probs = ret['probs'].cpu().detach().numpy().reshape([-1])
                    scores = ret['scores'].cpu().detach().numpy().reshape([-1])
                    symbols = ret['symbols'].cpu().detach().numpy()

                    best_trans, beam_trans = self.get_trans(probs, scores, symbols)
                    all_best_trans[original_idxs[i]] = best_trans + '\n'
                    all_beam_trans[original_idxs[i]] = beam_trans + '\n\n'

                    count += 1
                    if count % 100 == 0:
                        self.logger.info('  Translating line {}, average {} seconds/sent'.format(count, (time.time() - start) / count))

        model.train()

        open(best_trans_file, 'w').close()
        open(beam_trans_file, 'w').close()
        with open(best_trans_file, 'w') as ftrans, open(beam_trans_file, 'w') as btrans:
            ftrans.write(''.join(all_best_trans))
            btrans.write(''.join(all_beam_trans))

        self.logger.info('Done translating {}, it takes {} minutes'.format(input_file, float(time.time() - start) / 60.0))
