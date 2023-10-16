# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
import torch.nn.functional as F

# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerDecoderBasePrompt":
        return "TransformerDecoderPrompt"
    else:
        return module_name


class TransformerDecoderBasePrompt(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.dropout_presoftmax_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)
        
        self.encoder_prompts = None
        # prompts for each decoder layer
        if self.cfg.n_ensemble >= 1:
            if self.cfg.n_prompts != 0:
                self.encoder_prompts = nn.Parameter(torch.FloatTensor((self.cfg.n_ensemble + 1) * self.cfg.n_prompts, embed_dim))
                # print(self.encoder_prompts.size())
                nn.init.normal_(self.encoder_prompts, mean=0, std=embed_dim ** -0.5)

        if getattr(self.cfg,"feature_fusion_parameter", False): 
           self.feature_proj_list = nn.ModuleList([])
           if getattr(self.cfg,"feature_fusion_ln",False):
              self.fused_rep_ln_list = nn.ModuleList([])
              self.fused_rep_ln_list.extend([ LayerNorm(embed_dim, export=cfg.export)  for _ in range(self.cfg.n_ensemble-1)])
           if getattr(self.cfg,"feature_fusion_bn",False):
              self.fused_rep_bn_list = nn.ModuleList([])
              self.fused_rep_bn_list.extend([ nn.BatchNorm1d(embed_dim, affine=False)  for _ in range(self.cfg.n_ensemble-1)])
           
           fused_dim = embed_dim * 2 if not getattr(self.cfg, 'use_logit_dim', False) else len(dictionary)*2 
           # self.feature_proj_list.extend([Linear(fused_dim, 1, bias=True) for _ in range(self.cfg.n_ensemble-1)])
           #for _ in range(self.cfg.n_ensemble-1):
           #    param = Linear(fused_dim, embed_dim, bias=False)
           #    nn.init.normal_(param, mean=0, std=embed_dim ** -0.5)
           #    self.feature_proj_list.extend(param)
           self.feature_proj_list.extend([Linear(fused_dim, embed_dim, bias=False) for _ in range(self.cfg.n_ensemble-1)])

           self.kl_learned_weight = None
           self.ce_learned_weight = None

        if getattr(self.cfg,"kl_learned_weight",False):
           self.kl_learned_weight = nn.Parameter(torch.FloatTensor(self.cfg.n_ensemble, len(dictionary))) # 
        if getattr(self.cfg,"ce_learned_weight",False):
           self.ce_learned_weight = nn.Parameter(torch.FloatTensor(self.cfg.n_ensemble+1+self.cfg.n_ensemble-1)) # 
           nn.init.constant_(self.ce_learned_weight,5.0)
           # nn.init.xavier_uniform_(self.ce_learned_weight) 
           #nn.init.normal_(self.ce_learned_weight, mean=0, std=0.01)
        self.adaptive_kd_net = None
        if getattr(self.cfg,"adaptive_kd_network",None):
            # self.cfg.adaptive_kd_network_hidden_size:
            #     self.cfg.adaptive_kd_network_dropout:
            #         pass 
            self.adaptive_kd_net = nn.Sequential(
                nn.Linear(embed_dim, self.cfg.adaptive_kd_network_hidden_size), # [V, H]
                nn.Tanh(),
                nn.Dropout(p=self.cfg.adaptive_kd_network_dropout),
                nn.Linear(self.cfg.adaptive_kd_network_hidden_size, self.cfg.n_ensemble), # [H, M]
                nn.Softmax(dim=-1)
            )
            nn.init.normal_(self.adaptive_kd_net[0].weight, mean=0, std=0.01)
            nn.init.normal_(self.adaptive_kd_net[-2].weight, mean=0, std=0.01)
            
    def forward_adaptive_kd_net(self, features):
        # features.size() => [..., E]
        if self.adaptive_kd_net is None:
           return None  
        
        ouputs = self.adaptive_kd_net(features) # [..., 1]

        return ouputs

    def feature_fusion(self, feature1, feature2, proj_idx,decoder_hidden_padding=None):
        if feature2 is None:
           return feature1
        if self.cfg.mask_hidden_state:
           feature1 = feature1.masked_fill(decoder_hidden_padding,0.0)
           feature2 = feature2.masked_fill(decoder_hidden_padding,0.0)

        feature_concat =  torch.cat([feature1, feature2],dim=-1) # [B, T, 2E]
        fused_feature = self.feature_proj_list[proj_idx](feature_concat)
        if self.cfg.feature_fusion_ln:
           final_feature = self.fused_rep_ln_list[proj_idx](fused_feature + feature1 + feature2)
        else:
           final_feature = fused_feature + feature1 + feature2
        #weights = self.feature_proj_list[proj_idx](feature_concat) # [B, T, E]
        #weights = torch.sigmoid(weights) # [B, T, E]
        ##print(weights)
        #if self.cfg.feature_fusion_dropout:
        #   weights = self.dropout_module(weights)
        #final_feature = feature1 * weights  + feature2 * (1 - weights)
        #if self.cfg.feature_fusion_ln:
        #   final_feature = self.fused_rep_ln_list[proj_idx](final_feature)
        #if getattr(self.cfg,'feature_fusion_bn', False):
        #   #print('final_feature.size() => ',final_feature.size())
        #   final_feature = final_feature.reshape(-1,final_feature.size(-1))
        #   final_feature = self.fused_rep_bn_list[proj_idx](final_feature)
        #   final_feature = final_feature.reshape(feature1.size(0),-1,final_feature.size(-1))
        return final_feature

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        num_forward_decoder_layer =6,
        p=None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            num_forward_decoder_layer=num_forward_decoder_layer,
            p=p,
        )

        if not features_only and not self.training:
            if getattr(self.cfg,"layer_to_decode",6) != 6:
              #print('layer-to-decode: ',self.cfg.layer_to_decode)
              x = extra['inner_states'][self.cfg.layer_to_decode - 1].transpose(0,1)
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        num_forward_decoder_layer = 6,
        p=None
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            num_forward_decoder_layer=num_forward_decoder_layer,
            p=p,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        num_forward_decoder_layer=6,
        p=None
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            #alignment_layer = self.num_layers - 1
            alignment_layer = num_forward_decoder_layer

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        
        # print('prev_output_tokens.size()',prev_output_tokens.size())
        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
        
        #print('1 positions.size() => ',positions.size())
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        #print('2 positions.size() => ', positions.size())
        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x,p)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
       
        #print('x.size() => ',x.size())
        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        first_cat = True
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = []
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
               self_attn_mask = self.buffered_future_mask(x)
            else:
               self_attn_mask = None
            
            # if idx >= self.num_layers - self.cfg.n_ensemble - 1:
            if idx >= self.num_layers - self.cfg.n_ensemble -1: # and idx <= 2 + self.cfg.n_ensemble:
                # enc and padding_mask
                encoder_states = enc # [T x B x C]
                encoder_padding_mask = padding_mask # [B x T]
                if self.encoder_prompts is not None:
                    prompt_idx = idx - self.num_layers + self.cfg.n_ensemble + 1
                    # print('prompt_idx: ',prompt_idx)

                    prompts = self.encoder_prompts[self.cfg.n_prompts * prompt_idx: self.cfg.n_prompts * (prompt_idx + 1),:] # K x C
                    prompts = prompts.view(prompts.size(0),1,prompts.size(-1)) # K x 1 x C

                    prompt_padding_mask = torch.zeros(size=(encoder_padding_mask.size(0),prompts.size(0))).to(encoder_padding_mask) # B x K 
                    expand_prompts = prompts.repeat(1, encoder_padding_mask.size(0) ,1) # K x B x C
                    #prompt_padding_mask = torch.zeros(size=(self_attn_padding_mask.size(0),prompts.size(0))).to(self_attn_padding_mask) # B x K
                    #expand_prompts = prompts.repeat(1, encoder_padding_mask.size(0) ,1) # K x B x C
                   
                    #self_attn_padding_mask = torch.cat([prompt_padding_mask,self_attn_padding_mask], dim=1) # B x (K+T)                     
                    
                    if getattr(self.cfg, "prompt_position", "ahead") == 'ahead':
                       encoder_padding_mask = torch.cat([prompt_padding_mask,encoder_padding_mask],dim=1) # B x (K+T)
                       encoder_states = torch.cat([expand_prompts,encoder_states],dim=0) # (K+T) x B x C
                        
                    elif getattr(self.cfg, "prompt_position", "ahead")  == 'behind':
                       encoder_padding_mask = torch.cat([encoder_padding_mask, prompt_padding_mask],dim=1) # B x (T+K)
                       encoder_states = torch.cat([encoder_states, expand_prompts],dim=0) # (T+K) x B x C  
                       #print('behind')
                    elif getattr(self.cfg, "prompt_position", "ahead")  == 'ahead_and_behind':
                       encoder_padding_mask = torch.cat([prompt_padding_mask[:,:self.cfg.n_prompts//2], encoder_padding_mask, prompt_padding_mask[:,self.cfg.n_prompts//2:]],dim=1) # B x (K/2 + T + K/2)
                       encoder_states = torch.cat([expand_prompts[:self.cfg.n_prompts//2, :, :], encoder_states, expand_prompts[self.cfg.n_prompts//2:, :, :]],dim=0) # (K/2 + T + K/2) x B x C
                       #print('ahead_and_behind')
                    #encoder_padding_mask = torch.cat([prompt_padding_mask,encoder_padding_mask],dim=1) # B x (K+T)
                    #encoder_states = torch.cat([expand_prompts,encoder_states],dim=0) # (K+T) x B x C
                    #if self_attn_padding_mask is not None:
                    #      prompt_padding_mask = torch.zeros(size=(self_attn_padding_mask.size(0),prompts.size(0))).to(self_attn_padding_mask)
                    #      self_attn_padding_mask = torch.cat([prompt_padding_mask,self_attn_padding_mask], dim=1) # B x (K+T)   
                    #if first_cat:
                    #   x = torch.cat([expand_prompts,x],dim=0) 
                    #   first_cat = False
                    #   if self_attn_padding_mask is not None:
                    #      prompt_padding_mask = torch.zeros(size=(self_attn_padding_mask.size(0),prompts.size(0))).to(self_attn_padding_mask)
                    #      self_attn_padding_mask = torch.cat([prompt_padding_mask,self_attn_padding_mask], dim=1) # B x (K+T)
                    #   # print('first') 
                    #else:
                    #   x = x[expand_prompts.size(0):,:,:]
                    #   x = torch.cat([expand_prompts,x],dim=0)
                    #   # print('second')
            else:
                encoder_states = enc
                encoder_padding_mask = padding_mask   
            #print('training ...' if self.training else 'validating ...') 
            #print(x)
            #print('idx => ',idx)
            #print('x.size() => ',x.size())
            #print('='*100)
            #encoder_states = enc # [T x B x C]
            #encoder_padding_mask = padding_mask # [B x T]
            #print('idx ',idx)
            #print('num_forward_layer, ',)
            #print('self_attn_mask.size() ', self_attn_mask.size())
            #print('x.size() ',x.size())
            #print(num_forward_decoder_layer)
            #print(self_attn_padding_mask)
            #print('self_attn_padding_mask.size() ',self_attn_padding_mask.size())

            #if incremental_state is None and not full_context_alignment:
               #self_attn_mask = self.buffered_future_mask(x)
               #print('x,size() ',x.size())
               #print('self_attn_mask.size() ',self_attn_mask.size())
            #else:
               #self_attn_mask = None
            #print('idx ',idx)
            #print('self_attn_mask.size() ', self_attn_mask.size())
            #print('x.size() ',x.size())
            #print('self_attn_padding_mask.size() ',self_attn_padding_mask.size())
            #print('x.size() => ',x.size())
            #print('self_attn_padding_mask.size() => ',self_attn_padding_mask)
            #print('self_attn_mask.size() => ',self_attn_mask.size())
            x, layer_attn, _ = layer(
                x,
                encoder_states,
                encoder_padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                p=p,
            )
            #if idx <= self.num_layers - self.cfg.n_ensemble - 1:
               #print('transformer: ',idx)  
               #print('idx: ',self.num_layers - self.cfg.n_ensemble -1)
            inner_states.append(x)
    
            #if idx >= self.num_layers - self.cfg.n_ensemble - 1:
                #inner_states.append(x[self.cfg.n_prompts:])
                #inner_states.append(x)
                # prompt_idx = idx - self.num_layers + self.cfg.n_ensemble + 1
                # prompts = self.encoder_prompts[self.cfg.n_prompts * prompt_idx: self.cfg.n_prompts * (prompt_idx + 1),:] # K x C
                # prompts = prompts.view(prompts.size(0),1,prompts.size(-1)) # K x 1 x C

                # expand_prompts = prompts.repeat(1, encoder_padding_mask.size(0) ,1)

                # inner_states.append(torch.cat([expand_prompts,x],dim=0))

                #    inner_states.append(x[self.cfg.n_prompts:])
                # else:
                #    inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
            
            #if not self.training and idx == 0:
            #    break 
            if num_forward_decoder_layer is not None and idx == num_forward_decoder_layer:
               #print('num_forward_decoder_layer: ',num_forward_decoder_layer) 
               break
        # if wrd_decoder_layerelf.training:
        #    x = x[self.cfg.n_prompts:]
        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states,}

    def get_layer_wise_lprobs(self,
            decoder_output,
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,):
        log_probs_list = []
        inner_states = decoder_output[1]['inner_states'][::-1]

        n_layers = self.cfg.n_ensemble

        inner_states = inner_states[:n_layers]


        for idx,state in enumerate(inner_states):
            layer_out = self.output_layer(state).transpose(0,1)
     
            lprobs = self.get_normalized_probs((layer_out[:, -1:, :],),log_probs=log_probs, sample=sample)
            lprobs = lprobs[:, -1, :]
            log_probs_list.append(lprobs)
        avg_probs = torch.logsumexp(torch.stack(log_probs_list, dim=0), dim=0) - math.log(n_layers)

        return avg_probs

    def output_layer(self, features, exclude_prompt=True):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            out = self.output_projection(features)
            #if exclude_prompt:
            #    return out[:,self.cfg.n_prompts:,:]
            #else:
            #    return out  
            return out
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoderPrompt(TransformerDecoderBasePrompt):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )
