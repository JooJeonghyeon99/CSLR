# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys,logging
from argparse import Namespace
from typing import Any, Optional, Iterable

from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


from fairseq.tasks import register_task
from .dataset import AVHubertDataset_mvsr
from avhubert.hubert_pretraining import AVHubertPretrainingConfig, AVHubertPretrainingTask, LabelEncoder, LabelEncoderS2SToken


#★★★★★★ helper: convert OmegaConf cfg to plain dict for safe merging
def _task_cfg_to_dict(cfg: Optional[DictConfig]) -> dict:
    return {} if cfg is None else OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

@register_task("av_hubert_pretraining_mvsr", dataclass=AVHubertPretrainingConfig)
class AVHubertPretrainingTask_mvsr(AVHubertPretrainingTask):

    def load_dataset(self, split: str, **kwargs) -> None: # split = train, valid, test
        manifest = f"{self.cfg.data}/{split}.tsv" 
        dictionaries = [self.target_dictionary] if self.fine_tuning else self.dictionaries # 모델이 예측해야 할 target label의 dictionary
        # dictionaries = self.dictionaries #수정: 20251015
        # print(f'{len(dictionaries)=}') # 1
        pad_list = [dictionary.pad() for dictionary in dictionaries] # padding
        eos_list = [dictionary.eos() for dictionary in dictionaries] # eos token
        if not self.cfg.is_s2s: # for pretraining (비디오 유닛 만드니까)
            procs = [LabelEncoder(dictionary) for dictionary in dictionaries] # label encoder: discrete unit label을 dictionary에서 index로 mapping함
        else: # for finetuning : s2s is True
            logger.info(f"Using tokenizer")
            bpe_tokenizer = self.s2s_tokenizer #subword tokenizer
            procs = [LabelEncoderS2SToken(dictionary, bpe_tokenizer) for dictionary in dictionaries]
            
        # print('************', len(procs[0]))
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]
        image_aug = self.cfg.image_aug if split == 'train' else False
        noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if self.cfg.noise_wav is not None else None, eval(self.cfg.noise_snr) # fn은 파일목록 snr은 노이즈 비율
        noise_num = self.cfg.noise_num
        code_switching = getattr(self.cfg, "code_switching", None)
        logger.info(f"[CS][task] override.code_switching={code_switching}")
        self.datasets[split] = AVHubertDataset_mvsr( #위에서 설정한 값을 AVHubertDataset에 넣음
            manifest,
            code_switching=code_switching,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths,
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_sample_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_trim_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=True,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            stack_order_audio=self.cfg.stack_order_audio,
            skip_verify=self.cfg.skip_verify,
            image_mean=self.cfg.image_mean,
            image_std=self.cfg.image_std,
            image_crop_size=self.cfg.image_crop_size,
            image_aug=image_aug,
            modalities=self.cfg.modalities,
            is_s2s=self.cfg.is_s2s,
            noise_fn=noise_fn,
            noise_prob=self.cfg.noise_prob,
            noise_snr=noise_snr,
            noise_num=noise_num,
        )

#★★★(checkpoint config merge for code-switch decoding)
@register_task("av_hubert_pretraining_mvsr_cs", dataclass=AVHubertPretrainingConfig)
class AVHubertPretrainingTask_mvsr_cs(AVHubertPretrainingTask_mvsr):
    """Code-switching aware variant that reconciles checkpoint task config."""

    _OVERRIDE_KEYS: Iterable[str] = (
        "data", "label_dir", "tokenizer_bpe_model", "code_switching",
        "modalities", "noise_wav", "noise_prob", "noise_snr",
    )

    def __init__(self, cfg: AVHubertPretrainingConfig) -> None:
        super().__init__(cfg)
        self._runtime_cfg_cache = _task_cfg_to_dict(cfg)
        self._checkpoint_merged = False

    @classmethod
    def setup_task(cls, cfg: AVHubertPretrainingConfig, **kwargs):
        task = super().setup_task(cfg, **kwargs)
        if isinstance(task, cls):
            task._runtime_cfg_cache = _task_cfg_to_dict(cfg)
        return task

    def _merge_checkpoint_cfg(self, checkpoint_cfg: DictConfig) -> None:
        if checkpoint_cfg is None or self._checkpoint_merged:
            return
        saved_cfg = _task_cfg_to_dict(checkpoint_cfg)
        if not saved_cfg:
            return
        runtime_cfg = self._runtime_cfg_cache or _task_cfg_to_dict(self.cfg)
        merged_cfg = saved_cfg.copy()
        for key in self._OVERRIDE_KEYS:
            if key in runtime_cfg and runtime_cfg[key] is not None:
                merged_cfg[key] = runtime_cfg[key]
        OmegaConf.set_struct(self.cfg, False)
        for key, value in merged_cfg.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
        OmegaConf.set_struct(self.cfg, True)
        self._checkpoint_merged = True

    def load_dataset(self, split: str, task_cfg: DictConfig = None, **kwargs) -> None:
        if task_cfg is not None:
            self._merge_checkpoint_cfg(task_cfg)
        return super().load_dataset(split, **kwargs)

    def apply_checkpoint_cfg(self, checkpoint_cfg: DictConfig) -> None:
        self._merge_checkpoint_cfg(checkpoint_cfg)
