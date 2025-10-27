# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, logging
import contextlib
import tempfile
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Any, Iterable
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, FairseqEncoderDecoderModel, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING, OmegaConf, DictConfig

from einops import rearrange, repeat
from collections import OrderedDict
import random
import torch


from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
)

DBG = True if len(sys.argv) == 1 else False

if DBG:
    from avhubert.hubert import AVHubertModel
    from avhubert.decoder import TransformerDecoder
else:
    from avhubert.hubert import AVHubertModel
    from avhubert.decoder import TransformerDecoder

logger = logging.getLogger(__name__)


#★★★★★★ helper to align checkpoint task config with runtime overrides
PREFERRED_TASK_OVERRIDE_KEYS: Iterable[str] = (
    "data",
    "label_dir",
    "tokenizer_bpe_model",
    "code_switching",
    "modalities",
    "noise_wav",
    "noise_prob",
    "noise_snr",
)


def _apply_runtime_task_overrides(
    pretrained_task_cfg: Optional[DictConfig],
    runtime_task_cfg: Optional[DictConfig],
    keys: Iterable[str] = PREFERRED_TASK_OVERRIDE_KEYS,
) -> None:
    if pretrained_task_cfg is None or runtime_task_cfg is None:
        return
    runtime_dict = OmegaConf.to_container(runtime_task_cfg, resolve=True, enum_to_str=True)
    if not runtime_dict:
        return
    OmegaConf.set_struct(pretrained_task_cfg, False)
    try:
        for key in keys:
            if key in runtime_dict and runtime_dict[key] is not None:
                pretrained_task_cfg[key] = runtime_dict[key]
    finally:
        OmegaConf.set_struct(pretrained_task_cfg, True)




EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)


@dataclass
class encodercfg(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    input_modality: str = II("task.input_modality")
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=6, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=1024, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=4096, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=16, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_audio: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_length_image: int = field(default=10, metadata={"help": "mask length"})
    mask_prob_image: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})

    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type : str = field(default='same_other_seq', metadata={'help': 'type of selectig images, same_other_seq: replace masked span with span from another sequence, same_seq: repace masked span with span of the same sequence'})
    masking_type : str = field(default='input', metadata={'help': 'input or feature masking'})

    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
            "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "dropout probability for attention weights "
            "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
            "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})
    # code_switching: Optional[str] = field(default=None, metadata={'help': 'concatenate prompt'})  #cs용 인자 추가


@dataclass
class AVHubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to hubert model"}
    )
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout after transformer and before final projection"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
                    "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
                    "inside hubert model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
                    "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    masking_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None


@dataclass
class AVHubertCtcConfig(AVHubertAsrConfig):
    pass


@dataclass
class AVHubertSeq2SeqConfig(AVHubertAsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(
        default=6, metadata={"help": "num of decoder layers"}
    )
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings "
                    "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights "
                    "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN "
                    "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})
    progressive_masking: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    
    
class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: AVHubertAsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                cfg.w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        if runtime_task_cfg is None and hasattr(cfg, '_parent'):
            runtime_task_cfg = getattr(cfg._parent, 'task', None)
        #★★★(ensure pretrained hubert task uses current overrides)
        _apply_runtime_task_overrides(getattr(w2v_args, 'task', None), runtime_task_cfg)

        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = model.encoder.embedding_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }
        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class HubertEncoderWrapper(FairseqEncoder):
    def __init__(self, w2v_model):
        super().__init__(None)
        self.w2v_model = w2v_model

    def forward_(self, source, padding_mask, **kwargs):
        src ={}
        src['video'] = source
        src['audio'] = None
        w2v_args = {
            "source": src,
            "padding_mask": padding_mask,
        }

        x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask
        }


    def forward(self, source, padding_mask, **kwargs):
            w2v_args = {
                "source": source,
                "padding_mask": padding_mask,
            }

            x, padding_mask = self.w2v_model.extract_finetune(**w2v_args)


            return {
                "encoder_out": x,  # T x B x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask
            }

    def utut_forward(self, source, padding_mask):
            x, _ = self.w2v_model(
                source,
                padding_mask=padding_mask,
                layer=None
            ) 
            
            x = x.transpose(0, 1)
            
            return {
                "encoder_out": x,  # T x B x C
                "encoder_padding_mask": padding_mask,  # B x T
                "padding_mask": padding_mask
            }        

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        if 'decoder_global_lang_ids' in encoder_out and encoder_out['decoder_global_lang_ids'] is not None:
            encoder_out['decoder_global_lang_ids'] = encoder_out['decoder_global_lang_ids'].index_select(0, new_order)
        return encoder_out



@register_model("utut_seq2seq", dataclass=AVHubertSeq2SeqConfig)
class utut_seq2seq(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, tgt_dict, cfg, unit_encoder): #tgt_dict=출력 sequence token사전(transcript) (<s>, </s>, <pad>, <unk>), src_dict는 입력 sequence token사전(비디오 유닛, 단어)
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.freeze_finetune_updates = cfg.freeze_finetune_updates #finetuning 한거 frozen
        self.language_embedding = nn.Embedding(num_embeddings=5, embedding_dim=1024) #언어 5개 index로 받아서 1024 벡터로 변환해서 encoder 입력에 넣어줌
        self.ctc_proj = nn.Linear(1024, len(tgt_dict)) #ctc 학습용 projection layer
        self.unit_encoder = unit_encoder #pretraining용
        self.lang_tokens = ["<en>", "<it>", "<fr>", "<es>", "<pt>"]
        
        src_pth = os.path.dirname(os.path.realpath(__file__))
        utut_state_dict =torch.load(f'{src_pth}/pretrained_models/unit_pretrained/unit_pretrained.pt')['model']
        encoder_state_dict = {}
        decoder_state_dict = {}
        language_embedding_dict = {}
        for key in utut_state_dict.keys():
            if 'encoder' in key and 'decoder' not in key:
                encoder_state_dict[key[8:]] = utut_state_dict[key]
            elif 'decoder' in key:
                decoder_state_dict[key[8:]] = utut_state_dict[key]
            elif 'language_embedding' in key:
                language_embedding_dict[key[-6:]] = utut_state_dict[key] # 언어들
        self.unit_encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict, strict=False)
        self.language_embedding.load_state_dict(language_embedding_dict) # Pretraining에서 학습된 언어 embedding weight만 따로 불러와서 로드하고 사용

        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        
        src_pth = os.path.dirname(os.path.realpath(__file__))
        mavhubert_pth = f'{src_pth}/pretrained_models/mavhubert/mavhubert.pt' #미리 학습된 가중치를 읽으면서 w2v_args 인자를 가져옴
        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                mavhubert_pth, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        runtime_task_cfg = getattr(task, 'cfg', None)
        #★★★(pretrained task inherits fine-tuning overrides)
        _apply_runtime_task_overrides(getattr(w2v_args, 'task', None), runtime_task_cfg)
        if state is not None and 'cfg' in state and hasattr(state['cfg'], 'task'):
            _apply_runtime_task_overrides(state['cfg'].task, runtime_task_cfg)

        task_pretrain = tasks.setup_task(w2v_args.task) # pretraining할 때 사용한 task 설정
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model) # finetuning + inference 용

        encoder = HubertEncoderWrapper(encoder_) # finetuniong + inference 용
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            encoder.w2v_model.load_state_dict(state["model"], strict=False)

        encoder.w2v_model.remove_pretraining_modules()

        transformer_enc_cfg = encodercfg
        transformer_enc_cfg.encoder_layers = 6
        unit_encoder_ = TransformerEncoder(transformer_enc_cfg) # pretraining용 encoder
        unit_encoder = HubertEncoderWrapper(unit_encoder_) # pretraining용 encoder
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim): # token embedding 해주는거
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder, tgt_dict, cfg, unit_encoder) #자기 자신 class의 constructor
    
    def forward(self, **kwargs): # 전체 모델 inference 때 사용되는 함수 #**kwargs = net_input임 (dictionary)
            ft = self.freeze_finetune_updates <= self.num_updates #10000 전까진 finetuning 안함
            with torch.no_grad() if not ft else contextlib.ExitStack():
                output = self.encoder(**kwargs) 
            B, T, D = output['encoder_out'].size()
            
            # # -----------------------------------------
            # #    Concatenate prompt (mean embedding) 
            # # -----------------------------------------
            # lang_ids = torch.tensor(kwargs['languages'], device=output['encoder_out'].device, dtype=torch.long)
            # if lang_ids.dim() == 1:
            #     lang_embed = self.language_embedding(lang_ids) #[B, D]
            # else:
            #     concat_embed = self.language_embedding(lang_ids) #[B, K, D] = [batch size, number of language, embedding dimension]
            #     lang_embed = concat_embed.mean(dim=1) # 언어 개수에 대한 평균값 [B, D] = [batch size, embedding dimension]
            
            # logger.info(f"{torch.tensor(kwargs['languages'], dtype=int)=}")
            
            # lang_embed = lang_embed.unsqueeze(1).repeat(1, T, 1)
            # output['encoder_out'] += lang_embed
            # -----------------------------------------
            #   Hard gating (frame-wise) if available
            #   Fallback to global/mean embedding
            # -----------------------------------------
            if 'lang_frame_ids' in kwargs and kwargs['lang_frame_ids'] is not None:
                # lang_frame_ids: [B, T] with values in {0(en),1(it),2(fr),3(es),4(pt)}
                lfi = kwargs['lang_frame_ids'].to(output['encoder_out'].device).long()
                # [B, T, D]
                lang_embed = self.language_embedding(lfi)
            else:
                # 기존 전역 임베딩 경로 (단일 언어 혹은 여러 언어 평균)
                lang_ids = torch.tensor(kwargs['languages'], device=output['encoder_out'].device, dtype=torch.long)
                if lang_ids.dim() == 1:
                    lang_embed = self.language_embedding(lang_ids)  # [B, D]
                else:
                    concat_embed = self.language_embedding(lang_ids)  # [B, K, D]
                    lang_embed = concat_embed.mean(dim=1)            # [B, D]
                lang_embed = lang_embed.unsqueeze(1).repeat(1, T, 1)  # [B, T, D]

            output['encoder_out'] += lang_embed
            output = self.unit_encoder.utut_forward(output['encoder_out'], kwargs['padding_mask']) # unit을 encoder에 통과시켜서 contextualized representation 얻음
            
            output['decoder_lang_embed'] = lang_embed
            
            decoder_out = self.decoder(prev_output_tokens=kwargs['prev_output_tokens'], encoder_out=output)
            return decoder_out, output


    def extract_encoder_output(self, net_input): # extract output from encoder-only
        output = self.encoder(**net_input) #net_input을 dict 형태로 받음
        B, T, D = output['encoder_out'].size() # [배치, frame수, hidden dim]
        # # -----------------------------------------
        # #             Concatenate prompt 
        # # -----------------------------------------
        # lang_ids = torch.tensor(net_input['languages'], device=output['encoder_out'].device, dtype=torch.long)
        # if lang_ids.dim() == 1:
        #     lang_embed = self.language_embedding(lang_ids) #[B, D]
        # else:
        #     concat_embed = self.language_embedding(lang_ids) #[B, K, D] = [batch size, number of language, embedding dimension]
        #     lang_embed = concat_embed.mean(dim=1) # 언어 개수에 대한 평균값 [B, D] = [batch size, embedding dimension]

        # lang_embed = lang_embed.unsqueeze(1).repeat(1, T, 1)
        # output['encoder_out'] += lang_embed #두 모양이 같아져서 더해주면 됨
        # frame-wise 우선, 없으면 전역/평균
        if 'lang_frame_ids' in net_input and net_input['lang_frame_ids'] is not None:
            lfi = net_input['lang_frame_ids'].to(output['encoder_out'].device).long()
            lang_embed = self.language_embedding(lfi)  # [B,T,D]
        else:
            lang_ids = torch.tensor(net_input['languages'], device=output['encoder_out'].device, dtype=torch.long)
            if lang_ids.dim() == 1:
                lang_embed = self.language_embedding(lang_ids)  # [B, D]
            else:
                concat_embed = self.language_embedding(lang_ids)  # [B, K, D]
                lang_embed = concat_embed.mean(dim=1)            # [B, D]
            lang_embed = lang_embed.unsqueeze(1).repeat(1, T, 1)  # [B, T, D]

        output['encoder_out'] += lang_embed        
        
        output = self.unit_encoder.utut_forward(output['encoder_out'], net_input['padding_mask'])
        
        return output

    
    def get_ctc_target(self, sample):
        return sample["target"], sample["target_lengths"]

    def get_ctc_output(self, encoder_out, sample):
        en_out = encoder_out["encoder_out"]
        logits = self.ctc_proj(en_out)  # T x B x C
        out = utils.log_softmax(logits.float(), dim=-1)
        padding_mask = encoder_out["encoder_padding_mask"]
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
