import pathlib

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

## Define own callbacks here
def _resolve_checkpoint_root(PROJECT_PATH) -> pathlib.Path:
    """
    try to use cfg.paths.checkpoint_root first, if not, fallback to PROJECT_PATH/checkpoints
    """
    return (pathlib.Path(PROJECT_PATH) / "checkpoints").resolve()


def get_callbacks(cfg, PROJECT_PATH):
    callacks = []
    # ckpt_dir = pathlib.Path('/p/project/hai_denovo/Projects/debug/misato-ba/configs/runs')
    # ckpt_dir = pathlib.Path(HYDRA_CONFIG_PATH).joinpath("runs", cfg.name)
    ckpt_dir = _resolve_checkpoint_root(PROJECT_PATH)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    # Saves the top k checkpoints according to the test metric throughout
    # training.
    monitor_metric = "val/loss"
    monitor_mode = "max" if "f1" in monitor_metric.lower() else "min"
    print(f"monitor_metric: {monitor_metric}")
    print(f"monitor_mode: {monitor_mode}")
    # Clean metric name for filename
    metric_name = monitor_metric.replace("/", "_")
    checkpoint_filename = f"{{epoch}}-{{{metric_name}:.4f}}"
    print(f"checkpoint_filename: {checkpoint_filename}")
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best',#f"{cfg.name}" + "_{epoch}-{val_loss:.4f}",  # {epoch}-{val_loss:.4f}",
        every_n_epochs=1,
        monitor=monitor_metric, #f"{cfg.trainer.eval_metrics}",
        save_top_k=1,
        mode=monitor_mode #"min", #min 
    
        # save_last=True,  # Add this line to also save last epoch
    )
    callacks.append(ckpt)

    # Save last epoch checkpoint
    last_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=checkpoint_filename + "_last",#f"{cfg.name}" + "_{epoch}-{val_loss:.4f}" + "_last",
        save_last=True,
    )
    callacks.append(last_ckpt)

    # TODO add your own callbacks here

    return callacks