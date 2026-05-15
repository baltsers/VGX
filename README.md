# VGX

**VGX: Large-Scale Sample Generation for Boosting Learning-Based Software Vulnerability Analyses**

| | |
|---|---|
| Original artifact | <https://figshare.com/s/de1a7ca036bdc38d6a19> |
| Imported from | the publications page |
| Tool | `pubs2github` |


---

## Contents

The artifact contains 442 file(s) including Python, Shell scripts, Config files, and Documentation.

```
├── baselines
│   ├── getafix
│   │   ├── AST.py
│   │   ├── cluster.py
│   │   ├── const.py
│   │   ├── Hierarchical.py
│   │   ├── parse.py
│   │   ├── Pattern.py
│   │   └── testApplying.py
│   ├── graph2edit
│   │   ├── asdl
│   │   ├── common
│   │   ├── datasets
│   │   ├── edit_components
│   │   ├── scripts
│   │   ├── source_data
│   │   ├── __init__.py
│   │   └── exp_githubedits.py
│   ├── t5
│   │   ├── T5_vulgen_test_translate_final_tokenized2
│   │   ├── T5_beam1.py
│   │   ├── test.sh
│   │   └── train.sh
│   └── vulgen
│       ├── AST.py
│       ├── cluster.py
│       ├── CodeT5_raw_preds_final2_beam1.pkl
│       ├── const.py
│       ├── Hierarchical.py
│       ├── parse.py
│       ├── Pattern.py
│       └── testApplying.py
├── downstream_tasks
│   ├── detection
│   │   ├── devign
│   │   ├── ivdetect
│   │   └── linevul
│   ├── localization
│   │   ├── linevd
│   │   └── linevul
│   └── repair
│       ├── vrepair
│       └── vulrepair
├── VGX
│   ├── Contextualization
│   │   ├── checkpoint
│   │   ├── data
│   │   ├── dataset
│   │   ├── model
│   │   ├── trainer
│   │   └── main.py
│   └── Human-Knowledge-Enhanced-Edit-Pattern
│       ├── .Hierarchical.py.un~
│       ├── AST.py
│       ├── cluster.py
│       ├── const.py
│       ├── Contextualization_raw_preds_final2_beam1.pkl
│       ├── Contextualization_raw_preds_final2_beam1_no_ast.pkl
│       ├── Contextualization_raw_preds_final2_beam1_no_aug.pkl
│       ├── Contextualization_raw_preds_final2_beam1_no_flow.pkl
│       … (18 more items)
… (566 more items)
```

---

## Original `README.md` (from the upstream artifact)

# VGX: Large-Scale Sample Generation for Boosting Learning-Based Software Vulnerability Analyses
VGX is a new technique aimed for large-scale generation of high-quality vulnerability datasets. Given a normal program, VGX first identifies the code contexts in which vulnerabilities can be injected, using a customized source code Transformer featured with a new value-flow-based position encoding and pre-trained against new objectives particularly for learning code structure and context. Then, VGX materializes vulnerability-injection code editing in the identified contexts using patterns of such edits obtained from both historical fixes and human knowledge about real-world vulnerabilities. In this artifact, we provide the source code of VGX, the baselines compared, the generated dataset, as well as the downstream task tools augmented by the generated dataset.


## Package Structure
- `VGX.zip`: The source code, evaluation data, and results for VGX and its ablation study experiments.
    - `Contextualization/`: The source code, evaluation data, and results for VGX Step 1 Contextualization.
        - `data/`: The data used to for Contextualization.
        - `checkpoint/`: The trained models used to for Contextualization.
        - `main.py`: The main function for running contextualization.
    - `Human-Knowledge-Enhanced-Edit-Pattern/`: The source code, evaluation data, and results for VGX Step 2 Edit Pattern formation and vulnerability production.
        - `Contextualization_raw_preds_final2_beam1.pkl`: The contextualization results used for vulnerability production.
        - `Contextualization_raw_preds_final2_beam1_no_*.pkl`: The contextualization ablation study results used for vulnerability production.
        - `res_reg4_mutation.txt`: The experiment results on VGX vulnerability production.
        - `res_reg4_mutation_no_*.txt`: The ablation study experiment results on VGX vulnerability production.
        - `testApplying.py`: The main function for VGX vulnerability production.
        - `vulgen_test_final2.pkl`: The testing data for VGX vulnerability production.
- `baseline.zip`: The source code, evaluation data, and comparison results for VGX' baselines.
    - `vulgen/`: The source code, evaluation data, and results for VulGen evaluation.
        - `CodeT5_raw_preds_final_beam1.pkl`: The experiment output from the VulGen injection localization model on the testing set (used to localize the statement to inject vulnerability).
        - `testApplying.py` The script to test VulGen using the testing data and generate (possible) vulnerable functions.
        - `vulgen_test_final2.pkl`: The testing data for vulnerability production.
    - `getafix/`: The source code, evaluation data, and results for Getafix evaluation.
        - `testApplying.py` The script to test Getafix using the testing data and generate (possible) vulnerable functions.
        - `vulgen_test_final2.pkl`: The testing data for vulnerability production.
    - `T5/`: The source code, evaluation data, and results for Transformer-based injection localization and translation experiments.
        - `T5_vulgen_test_translate_final_tokenized/`: The teseting data for Transformer-based vulnerability generation baseline approach.
        - `T5_beam1.py`: The source code for T5 relevant models training and testing.
        - `train.sh`: The script to start training the T5 models.
        - `test.sh`: The script to test the trained T5 models for injection localization or Transformer-based vulnerability generation.
    - `graph2edit/`: The source code, evaluation data, and results for GNN-based vulnerability injection approach Graph2Edit.
        - `scripts/githubedits/`: The scripts to start training and testing the Graph2Edit model.
        - `source_data/githubedits/`: The training, validation, and testing data for Graph2Edit.
        - `exp_githubedits_runs/`: The training and testing outputs for Graph2Edit.
- `vgx_generated_full.zip`: The full dataset generated by VGX ready for use.
- `downstream_tasks.zip`: The downstream task tools augmented by the generated dataset and the respective experiments.
    - `detection/`: The source code, evaluation data, and augmentation results for DL-based vulnerability detection approach. 
        - `devign/`: The source code, evaluation data, and results for DL-based vulnerability detection approach Devign.
            - `devign-ori/`: The source code, data, and results before the augmentation.
            - `devign-aug/`: The source code, data, and results after the augmentation.
        - `linevul`: The source code, evaluation data, and results for DL-based vulnerability detection approach LineVul.
            - `linevul-ori/`: The source code, data, and results before the augmentation.
            - `linevul-aug/`: The source code, data, and results after the augmentation.
            - `data/`: The testing data used for evaluation.
        - `ivdetect/`: The source code, evaluation data, and results for DL-based vulnerability detection approach IVDetect.
            - `ivdetect-ori/`: The source code, data, and results before the augmentation.
            - `ivdetect-aug/`: The source code, data, and results after the augmentation.
            - `reveal_ivdetect.csv`: The testing data used for evaluation.
    - `localization/`: The source code, evaluation data, and augmentation results for DL-based vulnerability localization approach. 
        - `linevul`: The source code, evaluation data, and results for DL-based vulnerability localization approach LineVul.
            - `linevul-ori/`: The source code, data, and results before the augmentation.
            - `linevul-aug/`: The source code, data, and results after the augmentation.
            - `data/`: The testing data used for evaluation.
        - `linevd`: The source code, evaluation data, and results for DL-based vulnerability localization approach LineVD.
            - `linevd-ori/`: The source code, data, and results before the augmentation.
            - `linevd-aug/`: The source code, data, and results after the augmentation.
    - `repair/`: The source code, evaluation data, and augmentation results for DL-based vulnerability repair approach. 
        - `vrepair`: The source code, evaluation data, and results for DL-based vulnerability repair approach VRepair.
            - `vrepair-ori/`: The source code, data, and results before the augmentation.
            - `vrepair-aug/`: The source code, data, and results after the augmentation.
            - `data/`: The testing data used for evaluation.
        - `vulrepair`: The source code, evaluation data, and results for DL-based vulnerability repair approach VulRepair.
            - `vulrepair-ori/`: The source code, data, and results before the augmentation.
            - `vulrepair-aug/`: The source code, data, and results after the augmentation.
            - `data/`: The testing data used for evaluation.


## How to use

Please use the package structure to find the source code, evaluation data, and results for the corresponding contents described in the original paper. 




