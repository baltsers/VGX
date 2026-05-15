#!/bin/bash

#source activate structural_edits
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=/root/cwe_new/incremental_tree_edit:$PYTHONPATH

config_file=/root/cwe_new/incremental_tree_edit/exp_githubedits_runs/graph2iteredit.treediff_edit_encoder.json_branch_master_19e6ece.seed1000.20220105-190337/config.json

work_dir=/root/cwe_new/incremental_tree_edit/exp_githubedits_runs/graph2iteredit.treediff_edit_encoder.json_branch_master_19e6ece.seed1000.20220105-190337/ # TODO: replace FOLDER_OF_MODEL with your model dir

echo use config file ${config_file}
echo work dir=${work_dir}

mkdir -p ${work_dir}

# TODO: uncomment the test setting
# beam search
test_file=/root/cwe_new/incremental_tree_edit/source_data/githubedits/selected_testing_set_reveal_fix.jsonl
OMP_NUM_THREADS=1 python -m exp_githubedits decode_updated_data \
	  --cuda \
    --beam_size=1 \
    --evaluate_ppl \
    ${work_dir}/model.bin \
    ${test_file} 2>${work_dir}/test_reveal.log >${work_dir}/test_reveal.out  # dev_debug, dev, test, train_debug

# test ppl
#OMP_NUM_THREADS=1 python -m exp_githubedits test_ppl \
#	  --cuda \
#    --evaluate_ppl \
#    ${work_dir}/model.bin \
#    ${test_file} #2>${work_dir}/model.bin.decode_ppl.log


## csharp_fixer
#test_file=source_data/githubedits/csharp_fixers.jsonl
#scorer='default'
#OMP_NUM_THREADS=1 python -m exp_githubedits eval_csharp_fixer \
#	  --cuda \
#    --beam_size=1 \
#     --scorer=${scorer} \
#    ${work_dir}/model.bin \
#    ${test_file} 2>${work_dir}/model.bin.${filename}_csharp_fixer_${scorer}.log


## beam search on csharp_fixer with gold inputs
#test_file=source_data/githubedits/csharp_fixers.jsonl
#OMP_NUM_THREADS=1 python -m exp_githubedits decode_updated_data \
#	  --cuda \
#    --beam_size=1 \
#    ${work_dir}/model.bin \
#    ${test_file} 2>${work_dir}/model.bin.${filename}_csharp_fixer_gold.log


# collect edit vecs
##test_file=source_data/githubedits/csharp_fixers.jsonl
#test_file=source_data/githubedits/githubedits.test.jsonl
#OMP_NUM_THREADS=1 python -m exp_githubedits collect_edit_vecs \
#  --cuda \
#  ${work_dir}/model.bin \
#  ${test_file}

