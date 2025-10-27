current_pth=$(pwd)
fairseq_pth=$current_pth/fairseq
src_pth=$current_pth/src
# code_switching="fr-en"
code_switching="null"

data_pth=/mnt/aix7804/e-mvsr/labels/en
checkpoint_save_pth=/mnt/aix7804/e-mvsr/finetuning_en
PYTHONPATH=$fairseq_pth \
CUDA_VISIBLE_DEVICES=5,6,7 fairseq-hydra-train \
    --config-dir $src_pth/conf/ \
    --config-name finetuning.yaml \
    task.data=$data_pth \
    task.label_dir=$data_pth \
    dataset.train_subset=train \
    dataset.valid_subset=valid \
    task.tokenizer_bpe_model=$current_pth/spm1000/spm_unigram1000.model \
    model.w2v_path=$src_pth/pretrained_models/mavhubert/mavhubert.pt \
    hydra.run.dir=$checkpoint_save_pth \
    common.user_dir=$src_pth \
    task.code_switching="$code_switching" \
    checkpoint.restore_file=/mnt/aix7804/e-mvsr/finetuning_en/checkpoints/checkpoint_last.pt