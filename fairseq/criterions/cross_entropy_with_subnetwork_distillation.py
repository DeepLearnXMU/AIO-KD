# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#from fairseq import (
#    checkpoint_utils,
#    distributed_utils,
#    options,
#    quantization_utils,
#    tasks,
#    utils,
#)
import math
from multiprocessing import reduction
from statistics import mode

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass, field
import random
import numpy as np
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@dataclass
class CrossEntropyWithSubnetworkDistillationConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    kd_weight: float = field(
        default=2.5, metadata={"help": "weight for R-drop"}
    )
    ce_weight: float = field(
        default=1.0,
    )
    encoder_layer_max_idx: int = field(
        default=6,
    )
    encoder_layer_min_idx: int = field(
        default=1,
    )
    decoder_layer_max_idx: int = field(
        default=6,
    )
    decoder_layer_min_idx: int = field(
        default=1,
    )
    sample_student_number: int = field(
        default=1,
    )
    n_encoder_layer: int = field(
      default=6,
    )
    n_decoder_layer: int = field(
      default=6,
    )
    random_seed: int = field(
      default=64,
    )

    student_mutual_learning: str = field(
        default='none',
    )
    mutual_weight: float = field(
        default=2.5,
    )
    detach_threshold: float = field(
      default=999999,
    )
    sample_interval: int = field(
      default=1,
    )
    exclude_equal_student: bool = field(
      default=False,
    )
    uniform_sample: bool = field(
        default=False,
    )
    sample_include_teacher: bool = field(
       default=False,
    )
  
@register_criterion(
    "cross_entropy_with_subnetwork_distillation",
    dataclass=CrossEntropyWithSubnetworkDistillationConfig,
)

class CrossEntropyWithRdrop(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, kd_weight, ce_weight, encoder_layer_max_idx, encoder_layer_min_idx, decoder_layer_max_idx, decoder_layer_min_idx, sample_student_number, n_encoder_layer, n_decoder_layer, random_seed, student_mutual_learning, mutual_weight, detach_threshold, sample_interval, exclude_equal_student, uniform_sample, sample_include_teacher,):
        super().__init__(task, sentence_avg, label_smoothing)
        
        self.task = task
        self.kd_weight = kd_weight
        self.ce_weight = ce_weight

        self.encoder_layer_max_idx = encoder_layer_max_idx 
        self.encoder_layer_min_idx = encoder_layer_min_idx 

        self.decoder_layer_max_idx = decoder_layer_max_idx 
        self.decoder_layer_min_idx = decoder_layer_min_idx 

        self.sample_student_number =  sample_student_number
       
        self.n_encoder_layer = n_encoder_layer
        self.n_decoder_layer = n_decoder_layer
 
        self.random_seed = random_seed
        self.student_mutual_learning = student_mutual_learning

        self.mutual_weight = mutual_weight
        self.detach_threshold = detach_threshold
        self.sample_interval = sample_interval

        self.exclude_equal_student = exclude_equal_student
        self.uniform_sample = uniform_sample
        self.sample_include_teacher = sample_include_teacher

        # all candidate students from the teacher
        self.sample_space = [] if not self.sample_include_teacher else [(encoder_layer_max_idx-1, decoder_layer_max_idx-1)]
        tmp = [[(el, dl) for dl in range(decoder_layer_min_idx-1, el+1-exclude_equal_student)] for el in range(encoder_layer_min_idx-1, encoder_layer_max_idx)]
        for st in tmp:
            self.sample_space.extend(st)

        random.seed(random_seed)
        self.subnetwork_n_layer_pairs = []

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        probs = model.get_normalized_probs(net_output, log_probs=False)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), probs.view(-1,probs.size(-1)), target.view(-1)

    def get_net_output_list(self, model, sample):

        student_net_output_list = []
        teacher_net_output_list = []

        src_tokens, src_lengths, prev_output_tokens = sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens']

        tea_el_idx = self.n_encoder_layer - 1 
        tea_dl_idx = self.n_decoder_layer - 1
       
        # teacher
        tea_encoder_out = model.encoder(src_tokens, src_lengths=src_lengths,selected_layer_idx=tea_el_idx)
        tea_decoder_out = model.decoder(prev_output_tokens, encoder_out=tea_encoder_out, src_lengths=src_lengths,selected_layer_idx=tea_dl_idx)
        teacher_net_output_list.append(tea_decoder_out)
        
        if self.uniform_sample:
            if model.num_updates % self.sample_interval == 0:
                # print('Start Sampling... (num_updates: {})'.format(model.num_updates))
                self.subnetwork_n_layer_pairs.clear()
                for i in range(self.sample_student_number):
                    (stu_el_idx, stu_dl_idx) = random.choices(self.sample_space)[0]
                    stu_encoder_out = model.encoder(src_tokens, src_lengths=src_lengths,selected_layer_idx=stu_el_idx)
                    stu_decoder_out = model.decoder(prev_output_tokens, encoder_out=stu_encoder_out, src_lengths=src_lengths,selected_layer_idx=stu_dl_idx)
                    student_net_output_list.append(stu_decoder_out)
                    # print('stu_el_idx: {}, stu_dl_idx: {}'.format(stu_el_idx, stu_dl_idx))
                    # print('-'*100)
                    self.subnetwork_n_layer_pairs.append((stu_el_idx, stu_dl_idx))
            else:
                # print('Load sampled students ... (num_updates: {})'.format(model.num_updates))
                for i in range(self.sample_student_number):
                    (stu_el_idx, stu_dl_idx) = self.subnetwork_n_layer_pairs[i]
                    stu_encoder_out = model.encoder(src_tokens, src_lengths=src_lengths,selected_layer_idx=stu_el_idx)
                    stu_decoder_out = model.decoder(prev_output_tokens, encoder_out=stu_encoder_out, src_lengths=src_lengths,selected_layer_idx=stu_dl_idx)
                    student_net_output_list.append(stu_decoder_out)
                    # print('stu_el_idx: {}, stu_dl_idx: {}'.format(stu_el_idx, stu_dl_idx))
                    # print('-'*100)
        else:
            pass

        return teacher_net_output_list, student_net_output_list
    '''
        get the average cross-entropy losses of all candidate students
    '''
    def get_valid_loss(self, model, sample, reduce=True):
        
        src_tokens, src_lengths, prev_output_tokens = sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens']
        encoder_out = model.encoder(src_tokens, src_lengths=src_lengths,selected_layer_idx=self.n_encoder_layer - 1, return_all_hiddens=True) 

        ce_loss, nll_loss = None, None

        for idx_dl in range(self.decoder_layer_min_idx-1, self.decoder_layer_max_idx if not self.exclude_equal_student else self.decoder_layer_max_idx+1): 
            for idx_el in range(idx_dl if not self.exclude_equal_student else idx_dl + 1, self.encoder_layer_max_idx): 
                encoder_out['encoder_out'] = [encoder_out['encoder_states'][idx_el+1]]
                # print('eval: {}, {}'.format(idx_el, idx_dl))
                net_out = model.decoder(prev_output_tokens, encoder_out=encoder_out, src_lengths=src_lengths,selected_layer_idx=idx_dl)
                lprobs, probs, target = self.get_lprobs_and_target(model, net_out, sample)
                ce_loss_sub, nll_loss_sub = label_smoothed_nll_loss(
                      lprobs,
                      target,
                      self.eps,
                      ignore_index=self.padding_idx,
                      reduce=reduce,
                )
                ce_loss = ce_loss_sub if ce_loss is None else ce_loss + ce_loss_sub
                nll_loss = nll_loss_sub if nll_loss is None else nll_loss + nll_loss_sub
        return ce_loss, nll_loss 


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.training:

           teacher_net_output_list, student_net_output_list = self.get_net_output_list(model, sample)
        
           ce_loss, nll_loss, kd_loss, mutual_loss = self.compute_loss(model, teacher_net_output_list, student_net_output_list, sample, reduce=reduce)
           if mutual_loss is not None:
               loss = self.ce_weight  * ce_loss + kd_loss * self.kd_weight + mutual_loss * self.mutual_weight
           else:
               loss = self.ce_weight * ce_loss + kd_loss * self.kd_weight
        else:
           kd_loss = None
           loss, nll_loss = self.get_valid_loss(model, sample, reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "kd_loss": utils.item(kd_loss.data) if kd_loss is not None else utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output_teacher_list, net_output_student_list, sample, reduce=True):

        ce_loss = None 
        kd_loss = None 
        nll_loss = None

        stu_prob_st = []

        lprobs_tea, probs_tea, target = self.get_lprobs_and_target(model, net_output_teacher_list[0], sample)
        ce_loss_tea, nll_loss_tea = label_smoothed_nll_loss(
            lprobs_tea,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        target = target.view(target.size(0),1)


        stu_kd_loss_st = []
        stu_ce_loss_st = []
        stu_nll_loss_st = []
        stu_logit_st = []
        non_reduced_stu_ce_loss_st = []

        for idx, net_output_student in enumerate(net_output_student_list): 

            lprobs_stu, probs_stu, _ = self.get_lprobs_and_target(model, net_output_student, sample)
            stu_prob_st.append(probs_stu)
            ce_loss_stu, nll_loss_stu = label_smoothed_nll_loss(
                lprobs_stu,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
            
            # transfer the teacher's knowledge to each sample student
            # apply dynamic gradient detaching strategy
            ratio = (ce_loss_stu/ce_loss_tea).detach()
            # print('e{}d{},ratio:{}, threshold:{}'.format(self.subnetwork_n_layer_pairs[idx][0], self.subnetwork_n_layer_pairs[idx][1],ratio, self.detach_threshold))
            if ratio >= self.detach_threshold:
                stu_kd_loss = F.kl_div(lprobs_stu,probs_tea.detach(),reduction='none')
            else:
                stu_kd_loss = F.kl_div(lprobs_stu,probs_tea,reduction='none')

            # record the students' outputs
            stu_ce_loss_st.append(ce_loss_stu)
            stu_nll_loss_st.append(nll_loss_stu)
            stu_kd_loss_st.append(stu_kd_loss)

        mutual_loss = None
        if self.student_mutual_learning != 'none':
            for idx1,stu1_probs in enumerate(stu_prob_st):    
                for idx2, stu2_probs in enumerate(stu_prob_st[idx1+1:],start=idx1+1):
                    if mutual_loss is None:
                       if self.student_mutual_learning == 'full':
                          mutual_loss = (F.kl_div(torch.log(stu2_probs),stu1_probs,reduction='none')+F.kl_div(torch.log(stu1_probs),stu2_probs,reduction='none'))
                       elif self.student_mutual_learning == 'no_weight':
                          if stu_ce_loss_st[idx1] <= stu_ce_loss_st[idx2]:
                             mutual_loss = F.kl_div(torch.log(stu2_probs),stu1_probs,reduction='none')
                          else:
                             mutual_loss = F.kl_div(torch.log(stu1_probs),stu2_probs,reduction='none')
                       elif self.student_mutual_learning == 'detaching':
                          if stu_ce_loss_st[idx1] <= stu_ce_loss_st[idx2]:
                             if (stu_ce_loss_st[idx2]/stu_ce_loss_st[idx1]).detach() >= self.detach_threshold:
                                 mutual_loss = F.kl_div(torch.log(stu2_probs),stu1_probs.detach(),reduction='none')
                             else:
                                 mutual_loss = F.kl_div(torch.log(stu2_probs),stu1_probs,reduction='none')
                          else:
                             if (stu_ce_loss_st[idx1]/stu_ce_loss_st[idx2]).detach() >= self.detach_threshold:
                                 mutual_loss = F.kl_div(torch.log(stu1_probs),stu2_probs.detach(),reduction='none')
                             else:
                                 mutual_loss = F.kl_div(torch.log(stu1_probs),stu2_probs,reduction='none')
                    else:
                       if self.student_mutual_learning == 'full':
                          mutual_loss = mutual_loss + (F.kl_div(torch.log(stu2_probs),stu1_probs,reduction='none')+F.kl_div(torch.log(stu1_probs),stu2_probs,reduction='none'))
                       elif self.student_mutual_learning == 'no_weight':
                          if stu_ce_loss_st[idx1] <= stu_ce_loss_st[idx2]:
                             mutual_loss = mutual_loss + (F.kl_div(torch.log(stu2_probs),stu1_probs,reduction='none'))
                          else:
                             mutual_loss = mutual_loss + (F.kl_div(torch.log(stu1_probs),stu2_probs,reduction='none'))
                       elif self.student_mutual_learning == 'detaching':
                          if stu_ce_loss_st[idx1] <= stu_ce_loss_st[idx2]:
                             if (stu_ce_loss_st[idx2]/stu_ce_loss_st[idx1]).detach() >= self.detach_threshold:
                                mutual_loss = mutual_loss + (F.kl_div(torch.log(stu2_probs),stu1_probs.detach(),reduction='none'))
                             else:
                                mutual_loss = mutual_loss + (F.kl_div(torch.log(stu2_probs),stu1_probs,reduction='none'))
                          else:
                             if (stu_ce_loss_st[idx1]/stu_ce_loss_st[idx2]).detach() >= self.detach_threshold:
                                mutual_loss = mutual_loss + (F.kl_div(torch.log(stu1_probs),stu2_probs.detach(),reduction='none'))
                             else:
                                mutual_loss = mutual_loss + (F.kl_div(torch.log(stu1_probs),stu2_probs,reduction='none')) 
            
            # calculate the mutual learning loss 
            if self.student_mutual_learning == 'no_weight':
               mutual_loss = mutual_loss * 2.0 /(1.0 * self.sample_student_number * (self.sample_student_number - 1.))
            elif self.student_mutual_learning == 'full':
               mutual_loss = mutual_loss / (1.0 * self.sample_student_number * (self.sample_student_number - 1.))
            elif self.student_mutual_learning == 'detaching':
               mutual_loss = mutual_loss * 2.0 /(1.0 * self.sample_student_number * (self.sample_student_number - 1.))

        # calculate the knowledge distillation loss
        kd_loss = torch.stack(stu_kd_loss_st,dim=0).mean(dim=0)

        # calculate the cross-entropy losses for the teacher and students
        ce_loss = ce_loss_tea + torch.stack(stu_ce_loss_st,dim=0).sum()
        nll_loss = nll_loss_tea + torch.stack(stu_nll_loss_st,dim=0).sum()

        if reduce:
            target = target.view(target.size(0),1)
            kd_loss = kd_loss.view(-1,kd_loss.size(-1)) # [B*T, V] or [B*T, E]
            pad_mask = target.eq(self.padding_idx) # [B*T, 1]
            kd_loss.masked_fill_(pad_mask, 0.0)
            kd_loss = kd_loss.sum()
            
            if mutual_loss is not None:
               mutual_loss.masked_fill_(pad_mask, 0.0)
               mutual_loss = mutual_loss.sum()     

        return ce_loss, nll_loss, kd_loss, mutual_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )

        kd_loss_sum = utils.item(sum(log.get("kd_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))

        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
 
        metrics.log_scalar("kd_loss", kd_loss_sum)
        metrics.log_scalar('ntokens',ntokens) 
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
