import os
import time
from glob import glob
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
#from ray.tune import Analysis


chkpt = "/data/ynong/linevd/checkpoint/lightning_logs/version_18/checkpoints/epoch=9-step=10330.ckpt"
chkpt_res_path = "/data/ynong/linevd/checkpoint/lightning_logs/version_18/checkpoints/res.csv"
data = lvd.BigVulDatasetLineVDDataModule(
    batch_size=1024,
    nsampling_hops=2,
    gtype="pdg+raw",
    splits="default",
    feat="codebert",
)
# Load model and test
model = lvd.LitGNN()
model = lvd.LitGNN.load_from_checkpoint(chkpt, strict=False)
trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
trainer.test(model, data)
res = [0, "", model.res1vo,model.res2mt,model.res2f,model.res3vo,model.res2,model.lr]
# Save DF
mets = lvd.get_relevant_metrics(res)
res_df = pd.DataFrame(res)
#hparams = df[df.trial_id == res[0]][hparam_cols].to_dict("records")[0]
#res_df = pd.DataFrame.from_records([{**mets, **hparams}])
res_df.to_csv(chkpt_res_path, index=0)



