for target_lang in "en"

do
    # code_switching="fr-en"
    code_switching=null
    beam_size=20
    lenpen=0
    current_pth=$(pwd)
    echo "current_pth: $current_pth"
    fairseq_pth="$current_pth/fairseq"
    src_pth="$current_pth/src"
    # pretrained_model_pth="$src_pth/pretrained_models/finetuned/finetuned.pt"
    # pretrained_model_pth="/mnt/aix7804/e-mvsr/finetuning_en/checkpoints/checkpoint_best.pt"
    pretrained_model_pth="/mnt/aix7804/e-mvsr/finetuning_en/checkpoints/checkpoint_last.pt"
    label_pth="$current_pth/labels/$target_lang"
    echo "label_pth: $label_pth"
    results_save_pth=$src_pth/outputs/$target_lang
    
    PYTHONPATH=$fairseq_pth CUDA_VISIBLE_DEVICES=5,6,7 python -B $src_pth/infer_s2s.py \
        --config-dir $src_pth/conf/ \
        --config-name s2s_decode.yaml \
        dataset.gen_subset=test \
        common_eval.path="$pretrained_model_pth" \
        common_eval.results_path="$results_save_pth" \
        override.modalities="['video']" \
        common.user_dir="$src_pth" \
        generation.beam=$beam_size \
        generation.lenpen=$lenpen \
        override.data="$label_pth" \
        override.label_dir="$label_pth" \
        override.code_switching="$code_switching"

done