import os

import sastvd as svd
import sastvd.linevd.run as lvdrun
from ray import tune

os.environ["SLURM_JOB_NAME"] = "bash"
'''
config = {
    "hfeat": tune.choice([512]),
    "embtype": tune.choice(["codebert"]),
    "stmtweight": tune.choice([1]),
    "hdropout": tune.choice([0.3]),
    "gatdropout": tune.choice([0.2]),
    "modeltype": tune.choice(["gat2layer"]),
    "gnntype": tune.choice(["gat"]),
    "loss": tune.choice(["ce"]),
    "scea": tune.choice([0.5]),
    "gtype": tune.choice(["pdg+raw"]),
    "batch_size": tune.choice([1024]),
    "multitask": tune.choice(["linemethod"]),
    "splits": tune.choice(["default"]),
    "lr": tune.choice([1e-5]),
}
'''
config = {
    "hfeat": 512,
    "embtype": "codebert",
    "stmtweight": 1,
    "hdropout": 0.3,
    "gatdropout": 0.2,
    "modeltype": "gat2layer",
    "gnntype": "gat",
    "loss": "ce",
    "scea": 0.5,
    "gtype": "pdg+raw",
    "batch_size": 1024,
    "multitask": "linemethod",
    "splits": "default",
    "lr": 1e-5
}

samplesz = -1
lvdrun.train_linevd(config=config, max_epochs=130, samplesz=samplesz, savepath='./checkpoint')

'''
run_id = svd.get_run_id()
sp = svd.get_dir(svd.processed_dir() / f"raytune_best_{samplesz}" / run_id)
trainable = tune.with_parameters(
    lvdrun.train_linevd, max_epochs=130, samplesz=samplesz, savepath=sp
)

analysis = tune.run(
    trainable,
    resources_per_trial={"cpu": 2, "gpu": 0.5},
    metric="val_loss",
    mode="min",
    config=config,
    num_samples=180000,
    name="tune_linevd",
    local_dir=sp,
    keep_checkpoints_num=1,
    checkpoint_score_attr="min-val_loss",
)
'''
