python codebert_wordlevel_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --config_name=roberta-base \
    --do_test \
    --test_data_file=../data/fine_tune_data/patchdb_test_vrepair3.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 
