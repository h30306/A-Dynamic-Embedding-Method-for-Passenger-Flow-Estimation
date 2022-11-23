device=2
CUDA_VISIBLE_DEVICES=$device  python3.6 ./run_lm_finetuning.py \
        						--train_data_file=../../data/train_data_3loss.json \
        						--output_dir=./output/saveloss_20210725 \
        						--model_type=bert \
        						--model_name_or_path=bert-base-uncased \
        						--do_train \
        						--mlm \
        						--per_gpu_train_batch_size=640 \
        						--overwrite_cache \
        						--overwrite_output_dir \
        						--mlm_probability=0.15 \
                                --epoch=50 \
                                --num_attention_heads=8 \
                                --save_steps=150 \
                                --own_token=vocab.txt \
                                --hidden_size=768 \
                                
python3.6 ../../../RunningDoneLine.py --device=$device

