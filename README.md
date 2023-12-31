# Exploring All-In-One Knowledge Distillation Framework for Neural Machine Translation (AIO-KD)

Source code for the paper "Exploring All-In-One Knowledge Distillation Framework for Neural Machine Translation" (EMNLP2023 long paper). Notice that our AIO-KD is implemented based on [Fairseq tool](https://github.com/facebookresearch/fairseq) and our main modification compared to the original Fairseq is the addition of a code file [ "cross_entropy_with_subnetwork_distillation.py"](https://github.com/DeepLearnXMU/AIO-KD/blob/main/fairseq/criterions/cross_entropy_with_subnetwork_distillation.py).

## Requirements

- Python 3.7.0
- CUDA 11.7
- Pytorch 1.13.0
- Fairseq 0.12.2

## Citation
```
@inproceedings{miao2023exploring,
title={Exploring All-In-One Knowledge Distillation Framework for Neural Machine Translation},
  author={Miao, Zhongjian and Zhang, Wen and Su, Jinsong and Li, Xiang and Luan, Jian and Chen, Yidong and Wang, Bin and Zhang, Min},
  booktitle={Proc. of EMNLP},
  year={2023}
}
```

## Quickstart

Here, we take the IWSLT14 De-En translation task  as an example.

### Step1: Preprocess

```
# Download and clean the raw data
bash examples/translation/prepare-iwslt14.sh

# Preprocess/binarize the data
mkdir -p data-bin
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref /path/to/iwslt14_deen_data/train \
    --validpref /path/to/iwslt14_deen_data/valid \
    --testpref /path/to/iwslt14_deen_data/test \
    --destdir /path/to/data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### Step2: Training
- Hyper-parameters in AIO-KD
```
  sample_student=2
  encoder_layer_max_idx=6
  encoder_layer_min_idx=2
  decoder_layer_max_idx=6
  decoder_layer_min_idx=2
  kd_weight=5.5
  ce_weight=1.0
  sml_weight=0.5
  threshold=1.1
  ```
- The first training stage:
  ```
  mkdir -p /path/to/ckpts/iwslt14_de-en/stage1
  mkdir -p /path/to/ckpts/iwslt14_de-en/stage2
  fairseq-train /path/to/data-bin/iwslt14.tokenized.de-en \
    --task translation --arch  transformer_iwslt_de_en \
    --share-all-embeddings --optimizer adam --lr 0.0005 -s de -t en --label-smoothing 0.1 \
     --max-tokens 4096 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion cross_entropy_with_subnetwork_distillation --no-progress-bar --seed 64 \
    --encoder-layer-max-idx $encoder_layer_max_idx --encoder-layer-min-idx $encoder_layer_min_idx --n-encoder-layer $encoder_layer_max_idx \
    --decoder-layer-max-idx $decoder_layer_max_idx --decoder-layer-min-idx $decoder_layer_min_idx  --n-decoder-layer $decoder_layer_max_idx \
    --kd-weight $kd_weight --ce-weight $ce_weight  --sample-student-number $sample_student --uniform-sample \
    --no-epoch-checkpoints --detach-threshold $threshold \
    --max-update 300000 \
    --warmup-updates 4000 --warmup-init-lr 1e-07--adam-betas '(0.9,0.98)' \
    --save-dir /path/to/ckpts/iwslt14_de-en/stage1 \
    --no-epoch-checkpoints --fp16 --dropout 0.3
  ```
- The Second training stage:
  
  ```
  fairseq-train /path/to/data-bin/iwslt14.tokenized.de-en \
    --task translation --arch  transformer_iwslt_de_en  --share-all-embeddings \
    --optimizer adam --lr 0.0005 -s de -t en --label-smoothing 0.1 \
    --max-tokens 4096 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion cross_entropy_with_subnetwork_distillation --no-progress-bar --seed 64 \
    --encoder-layer-max-idx $encoder_layer_max_idx --encoder-layer-min-idx $encoder_layer_min_idx --n-encoder-layer $encoder_layer_max_idx \
    --decoder-layer-max-idx $decoder_layer_max_idx --decoder-layer-min-idx $decoder_layer_min_idx  --n-decoder-layer $decoder_layer_max_idx \
    --mutual-weight $sml_weight --kd-weight $kd_weight --ce-weight $ce_weight  --sample-student-number $sample_student  \
    --no-epoch-checkpoints --detach-threshold $threshold --uniform-sample \
    --max-update 300000 --student-mutual-learning no_weight \
    --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --adam-betas '(0.9,0.98)' --save-dir /path/to/ckpts/iwslt14_de-en/stage2 \
    --no-epoch-checkpoints --fp16 --dropout 0.3 \
    --finetune-from-model /path/to/ckpts/iwslt14_de-en/stage1/checkpoint_best.pt
  ```

### Step3: Evaluation

```
for el in 2 3 4 5 6
   do for dl in 2 3 4 5 6
         do
         echo "Decoding: ""encoder layer:"$el", decoder layer:"$dl
         # generate translations
         fairseq-generate /path/to/data-bin/iwslt14.tokenized.de-en \
	   --path /path/to/ckpts/iwslt14_de-en/stage2/checkpoint_best.pt \
           --encoder-layer-to-infer $el --decoder-layer-to-infer $dl  \
	   --beam 5 --remove-bpe > /path/to/ckpts/iwslt14_de-en/stage2/res-e${el}d${dl}.out
         # calculate BLEU score   
         bash scripts/compound_split_bleu.sh 
    done
done
```
