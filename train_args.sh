python3 scicap_sum_baseline.py \
    --num_train_epochs 3 \
    --model_name_or_path google/pegasus-arxiv \
    --output_dir baseline \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --train_file split_json_data_dir/train.json \
    --validation_file split_json_data_dir/val.json \
    --test_file split_json_data_dir/test.json \
    --dataset_name scicap_summary \
    --sortish_sampler \
    --max_source_length 512 \
    --max_target_length 100 \
    --load_best_model_at_end \
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --warmup_steps=500 \
    --learning_rate=5e-5 \
    --do_train \
    --do_eval \
    --do_predict \
    --text_column paragraph \
    --summary_column caption_no_index \
