# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = str(Path(__file__).resolve().parent / "conf")  #hydra 버전 안맞아서 str으로 감싸줌

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})
    code_switching: Optional[str] = field(default=None, metadata={'help': 'concatenate prompt'})  #cs용 인자 추가


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def main(cfg: DictConfig):
    
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}

def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize") #
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    models = [model.eval().cuda() for model in models]
    print('******************', saved_cfg.task)
    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)
    task.build_tokenizer(saved_cfg.tokenizer) #e-mvsr/fairseq/fairseq/tasks/fairseq_task.py에 들어있음
    task.build_bpe(saved_cfg.bpe)
    logger.info(cfg)
    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    dictionary = task.target_dictionary

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    
    #★★★(mirror noise overrides into checkpoint config)
    if OmegaConf.is_config(getattr(saved_cfg, "task", None)):
        OmegaConf.set_struct(saved_cfg.task, False)
        saved_cfg.task.noise_prob = cfg.override.noise_prob
        saved_cfg.task.noise_snr = cfg.override.noise_snr
        saved_cfg.task.noise_wav = cfg.override.noise_wav
        OmegaConf.set_struct(saved_cfg.task, True)
        
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data  #label path
        
        #★★★(keep saved task config paths in sync)
        if OmegaConf.is_config(getattr(saved_cfg, "task", None)):
            OmegaConf.set_struct(saved_cfg.task, False)
            saved_cfg.task.data = cfg.override.data
            OmegaConf.set_struct(saved_cfg.task, True)
            
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
        
        #★★★(propagate label_dir override)
        if OmegaConf.is_config(getattr(saved_cfg, "task", None)):
            OmegaConf.set_struct(saved_cfg.task, False)
            saved_cfg.task.label_dir = cfg.override.label_dir
            OmegaConf.set_struct(saved_cfg.task, True)

        
     # --- [CS] propagate code_switching from CLI override -> task configs BEFORE load_dataset() ---
    cs = getattr(cfg.override, "code_switching", None)
    if cs:
        # 1) runtime task config
        try:
            task.cfg.code_switching = cs
        except Exception:
            pass
        # 2) saved task config (used by load_dataset(task_cfg=...))
        try:
            #★★★(propagate code-switching override)
            if OmegaConf.is_config(getattr(saved_cfg, "task", None)):
                OmegaConf.set_struct(saved_cfg.task, False)
                saved_cfg.task.code_switching = cs
                OmegaConf.set_struct(saved_cfg.task, True)
        except Exception:
            pass
        # logger.info(f"[CS] override='{cs}' -> task.cfg.code_switching={getattr(task.cfg,'code_switching',None)}; "
                    # f"saved_cfg.task.code_switching={getattr(saved_cfg.task,'code_switching',None)}")

    if hasattr(task, "apply_checkpoint_cfg"):
        #★★★(reapply runtime overrides from checkpoints before loading dataset)
        task.apply_checkpoint_cfg(saved_cfg.task)

    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)  # load data
    
    
    # --- [CS] dataset-side confirmation log ---
    ds = task.datasets[cfg.dataset.gen_subset]
    # logger.info(f"[CS] dataset.code_switching={getattr(ds,'code_switching',None)}, "
    #             f"ds.cs_pair_ids={getattr(ds,'cs_pair_ids',None)}")


    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    if cfg.generation.match_source_len:
        logger.warning(
            "The option match_source_len is not applicable to speech recognition. Ignoring it."
        )
    gen_timer = StopwatchMeter()
    extra_gen_cls_kwargs = {
        "lm_model": lms[0],
        "lm_weight": cfg.generation.lm_weight,
    }
    cfg.generation.score_reference = False  #
    save_attention_plot = cfg.generation.print_alignment is not None
    cfg.generation.print_alignment = None  #
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # -----------------------------------------
    # Concatenate Prompt: +override.code_switching='fr-en' 해주려고
    # 논문에서 2개 이상 언어 토큰을 디코더 프롬프트에 한번에 (concat) 넣어 CS 하도록 함
    # -----------------------------------------
    def build_lang_prompt_tensor(dictionary, code_switching, device, dtype, batch_size, add_bos=True):
        # Purpose 'fr-en' -> ['fr','en'] -> ['<fr>','<en>']
        lang_list = [lang.strip() for lang in code_switching.split('-') if lang.strip()]
        tokens = []
        if add_bos:  # <s> dictionary에서는 bos로 되어있는거 가져옴 (fairseq.data.Dictionary에 되어있음)
            tokens.append(dictionary.bos())
        for lang_token in lang_list:
            lang_token_symbol = f"<{lang_token}>"
            idx = dictionary.index(lang_token_symbol)
            if idx == dictionary.unk():
                raise RuntimeError(f"Language token '{lang_token_symbol}' not found in target dict..")
            tokens.append(idx) #토큰 더해줌
        # unsqueeze(0)으로 [1, K] -> repeat으로 row를 batch_size만큼 늘림 [B, K]: 문장 당 batch만큼 prompt 가짐
        prompt = torch.tensor(tokens, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)  # [B, K]
        return prompt

    def decode_fn(x):
        symbols_ignore = get_symbols_to_strip_from_output(generator)
        symbols_ignore.add(dictionary.pad())
        
        #-----------------lang token remove------------
        lang_tags = ["<en>", "<it>", "<fr>", "<es>", "<pt>"]
        for tag in lang_tags:
            idx = dictionary.index(tag)
            if idx != dictionary.unk():
                symbols_ignore.add(idx)
                
                
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return task.datasets[cfg.dataset.gen_subset].label_processors[0].decode(x, symbols_ignore)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}

    for sample in progress:

        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        
        prefix_tokens = None  #위치 옮김

        # -----------------------------------------
        #          Concatenate prompt 넣기
        # -----------------------------------------
        # if cfg.override.code_switching:
        #     batch_size = sample["id"].size(0)  # id: batch에 들어온 samples의 idx모음, 즉 batch에 sample 몇 개 들어있는지?
        #     device = models[0].decoder.embed_tokens.weight.device
        #     prefix_tokens = build_lang_prompt_tensor( # prompt를 prefix_token으로 전달
        #         dictionary, cfg.override.code_switching, device, torch.long, batch_size, add_bos=False 
        #     )
        #     syms = [task.target_dictionary.symbols[t] for t in prefix_tokens[0].tolist()] #삐띠니가 준 logger: prefix_tokens[0] (언어 토큰) -> target_dictionary에서 각 step별 symbols
        #     logger.info(f"[CS-PREFIX] {cfg.override.code_switching} -> {syms}") #삐띠니가 준 logger
        #     # ----------------------------------------------
        #     #         encoder에서 변경한거 적용(추가)
        #     # ----------------------------------------------
        #     lang_tokens = [lang.strip() for lang in cfg.override.code_switching.split('-') if lang.strip()]
            
        #     lang_idx_list = {"en":0, "it":1, "fr":2, "es":3, "pt":4}
        #     lang_ids = [lang_idx_list[name] for name in lang_tokens]

        #     lang_tensor = torch.tensor([lang_ids], device=device).repeat(batch_size, 1)  # [B,K]
        #     sample["net_input"]["languages"] = lang_tensor

                        
        # elif cfg.generation.prefix_size > 0:
        #     prefix_tokens = sample["target"][:, :cfg.generation.prefix_size]
                
        if cfg.override.code_switching:
            batch_size = sample["id"].size(0)  # id: batch에 들어온 samples의 idx모음, 즉 batch에 sample 몇 개 들어있는지?
            device = models[0].decoder.embed_tokens.weight.device
            # ----------------------------------------------
            #         encoder에서 변경한거 적용(추가)
            # ----------------------------------------------
            lang_tokens = [lang.strip() for lang in cfg.override.code_switching.split('-') if lang.strip()]
            
            lang_idx_list = {"en":0, "it":1, "fr":2, "es":3, "pt":4}
            lang_ids = [lang_idx_list[name] for name in lang_tokens]

            lang_tensor = torch.tensor([lang_ids], device=device).repeat(batch_size, 1)  # [B,K]
            sample["net_input"]["languages"] = lang_tensor

        elif cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]


        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens= prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            ref_sent = decode_fn(sample['target'][i].int().cpu())
            result_dict['ref'].append(ref_sent)
            best_hypo = hypos[i][0]['tokens'].int().cpu()
            hypo_str = decode_fn(best_hypo)
            result_dict['hypo'].append(hypo_str)
            logger.info(f"\nREF:{ref_sent}\nHYP:{hypo_str}\n")

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += sample["nsentences"] if "nsentences" in sample else sample["id"].numel()

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Recognized {:,} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1.0 / gen_timer.avg
        )
    )

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)

    n_err, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)

    wer = 100 * n_err / n_total
    wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
    with open(wer_fn, "w") as fo:
        fo.write(f"WER: {wer}\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
        fo.write(f"{yaml_str}")

    logger.info(f"WER: {wer}%")
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import get_args  # pylint: disable=import-outside-toplevel
        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
