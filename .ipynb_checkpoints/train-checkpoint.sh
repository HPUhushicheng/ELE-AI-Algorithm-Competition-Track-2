#代码中的路径问题还请酌情修改，有问题请及时联系我们

#首先使用预训练qwen2.5-vl-7b预训练模型进行lora微调训练，合并适配器之后再进行推理预测，适配器保存在'saves/Qwen2.5-VL-7B-Instruct/lora/fine-tuning-results'
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /root/autodl-tmp/Qwen2.5-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir /root/LLaMA-Factory/data \
    --dataset train-aug \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 300 \
    --warmup_steps 10 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-VL-7B-Instruct/lora/Qwen2.5-VL-7B-Instruct-fine-tuning \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target all \
    --val_size 0.1 \
    --eval_strategy steps \
    --eval_steps 300 \
    --per_device_eval_batch_size 4




#接着要将预训练模型和 LoRA 适配器合并导出成一个模型，最终的模型文件保存在'/root/autodl-tmp/qwen2.5-vl-7b-sft'中   
 
llamafactory-cli export /root/autodl-tmp/project/code/qwen2_5vl_lora_sft.yaml


#同样提取高风险样本和已有标签进行lora微调得到qwen2.5-vl-3b-sft-high-risk模型
