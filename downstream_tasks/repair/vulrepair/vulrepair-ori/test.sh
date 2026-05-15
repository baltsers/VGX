python -u vulrepair_main.py \
    --output_dir=./saved_models \
    --model_name=model.bin \
    --tokenizer_name=MickyMike/VulRepair \
    --model_name_or_path=MickyMike/VulRepair \
    --do_test \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --num_beams=50 \
    --eval_batch_size 1
