import sys
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

from transfer_model import TransferModel
from datasets import FireProtDataset, MegaScaleDataset, ComboDataset
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from datetime import datetime
import os
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
import math


def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
    }


class TransferModelPL(pl.LightningModule):
    """Class managing training loop with pytorch lightning"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransferModel(cfg)

        self.encoder_lr_init = cfg.training.encoder_lr_init
        self.encoder_lr_final = cfg.training.encoder_lr_final
        self.edge_lr_init = cfg.training.edge_lr_init
        self.edge_lr_final = cfg.training.edge_lr_final
        self.head_lr_init = cfg.training.head_lr_init
        self.head_lr_final = cfg.training.head_lr_final
        self.light_attention_lr_init = cfg.training.light_attention_lr_init
        self.light_attention_lr_final = cfg.training.light_attention_lr_final


        self.mpnn_learn_rate = cfg.training.mpnn_learn_rate if 'mpnn_learn_rate' in cfg.training else None
        self.lr_schedule = cfg.training.lr_schedule if 'lr_schedule' in cfg.training else False

        # set up metrics dictionary
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "ddG"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):

        assert len(batch) == 1
        mut_pdb, mutations, disto, bin_to_sum = batch[0]
        pred, _ = self(mut_pdb, mutations, disto, bin_to_sum)

        ddg_mses = []
        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                ddg_mses.append(F.mse_loss(out["ddG"], mut.ddG))
                for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

        loss = 0.0 if len(ddg_mses) == 0 else torch.stack(ddg_mses).mean()
        on_step = False
        on_epoch = not on_step

        output = "ddG"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                        batch_size=len(batch))
        if loss == 0.0:
            return None
        return loss
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        for pg, name in zip(optimizer.param_groups, self.group_names):
            self.log(f"lr/{name}", pg["lr"], on_epoch=True)
    
    #def on_fit_start(self):
    #    optimizer = self.optimizers()
    #    for pg, name in zip(optimizer.param_groups, self.group_names):
    #        self.logger.experiment.add_scalar(
    #            f"lr/{name}", pg["lr"], global_step=0
    #        )
    
    def on_train_epoch_start(self):
        
        # Define progressive unfreeze epochs
        unfreeze_schedule = getattr(self.cfg.training, "unfreeze_epochs")
        # unfreeze_schedule[i] = epoch at which next encoder layer is unfrozen
        
        if self.current_epoch < unfreeze_schedule[1]:
            # Only first encoder layer
            self.freeze_all()
            self.enable_edge_embedding()
            self.enable_encoder_layers(max_layer=0)
        elif self.current_epoch < unfreeze_schedule[2]:
            # First two encoder layers
            self.freeze_all()
            self.enable_edge_embedding()
            self.enable_encoder_layers(max_layer=1)
        else:
            # All encoder layers
            self.freeze_all()
            self.enable_edge_embedding()
            self.enable_encoder_layers(max_layer=2)

        # Light attention + heads always trainable
        self.enable_light_attention_and_heads()
        
        self.print_trainable_params()

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        total_epochs = self.cfg.training.epochs

        lr_init = {
            "encoder": self.encoder_lr_init,
            "edge": self.edge_lr_init,
            "head": self.head_lr_init,
            "light_attention": self.light_attention_lr_init,
        }
        lr_final = {
            "encoder": self.encoder_lr_final,
            "edge": self.edge_lr_final,
            "head": self.head_lr_final,
            "light_attention": self.light_attention_lr_final,
        }

        param_groups = []
        group_names = []

        # ---- Edge-related params ----
        edge_params = []
        edge_params += self.model.prot_mpnn.features.edge_embedding.parameters()
        edge_params += self.model.prot_mpnn.features.norm_edges.parameters()
        edge_params += self.model.prot_mpnn.W_e.parameters()

        param_groups.append({
            "params": [p for p in edge_params if p.requires_grad],
            "lr": lr_init["edge"],
        })
        group_names.append("edge")

        # ---- Encoder layers ----
        for i, layer in enumerate(self.model.prot_mpnn.encoder_layers):
            params = [p for p in layer.parameters() if p.requires_grad]
            if not params:
                continue
            param_groups.append({
                "params": params,
                "lr": lr_init["encoder"],
            })
            group_names.append(f"encoder_{i}")

        # ---- Light attention ----
        if self.model.lightattn:
            param_groups.append({
                "params": self.model.light_attention.parameters(),
                "lr": lr_init["light_attention"],
            })
            group_names.append("light_attention")

        # ---- Heads ----
        param_groups.append({
            "params": self.model.both_out.parameters(),
            "lr": lr_init["head"],
        })
        group_names.append("both_out")

        param_groups.append({
            "params": self.model.ddg_out.parameters(),
            "lr": lr_init["head"],
        })
        group_names.append("ddg_out")

        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        # ---- ONE scheduler with ONE lambda per group ----
        def make_lambda(group_name):
            if "encoder" in group_name:
                key = "encoder"
            elif "edge" in group_name:
                key = "edge"
            elif "light_attention" in group_name:
                key = "light_attention"
            else:
                key = "head"

            l_init = lr_init[key]
            l_final = lr_final[key]

            def lr_lambda(epoch):
                lr = l_final + 0.5 * (l_init - l_final) * (
                    1 + math.cos(math.pi * epoch / total_epochs)
                )
                return lr / l_init

            return lr_lambda

        lr_lambdas = [make_lambda(name) for name in group_names]

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambdas
        )

        self.group_names = group_names

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }]

    #def configure_optimizers(self):
    #    if self.stage == 2: # for second stage, drop LR by factor of 10
    #        self.learn_rate /= 10.
    #        print('New second-stage learning rate: ', self.learn_rate)
#
    #    if not cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
    #        param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": self.mpnn_learn_rate}]
    #    else: # fully frozen MPNN
    #        param_list = []
#
    #    if self.model.lightattn:  # adding light attention parameters
    #        if self.stage == 2:
    #            param_list.append({"params": self.model.light_attention.parameters(), "lr": 0.})
    #        else:
    #            param_list.append({"params": self.model.light_attention.parameters()})
    #            #param_list.append({"params": self.model.light_attention.parameters(), "lr": self.learn_rate})
#
#
    #    mlp_params = [
    #        {"params": self.model.both_out.parameters()},
    #        {"params": self.model.ddg_out.parameters()}
    #        ]
#
    #    param_list = param_list + mlp_params
    #    opt = torch.optim.AdamW(param_list, lr=self.learn_rate)
#
    #    if self.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
    #        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5)
    #        return {
    #            'optimizer': opt,
    #            'lr_scheduler': lr_sched,
    #            'monitor': 'val_ddG_mse'
    #        }
    #    else:
    #        return opt
    
    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False


    def enable_edge_embedding(self):
        # Unfreeze edge embedding
        for p in self.model.prot_mpnn.features.edge_embedding.parameters():
            p.requires_grad = True

        # Unfreeze intermediate layers between edge_embedding and first encoder
        #for p in self.model.prot_mpnn.features.embeddings.linear.parameters():
        #    p.requires_grad = True
        for p in self.model.prot_mpnn.features.norm_edges.parameters():
            p.requires_grad = True
        for p in [self.model.prot_mpnn.W_e]:
            for param in p.parameters():
                param.requires_grad = True


    def enable_encoder_layers(self, max_layer: int):
        """
        Enable encoder layers [0 .. max_layer]
        """
        for i in range(max_layer + 1):
            for p in self.model.prot_mpnn.encoder_layers[i].parameters():
                p.requires_grad = True


    def enable_light_attention_and_heads(self):
        if self.model.lightattn:
            for p in self.model.light_attention.parameters():
                p.requires_grad = True

        for p in self.model.both_out.parameters():
            p.requires_grad = True

        for p in self.model.ddg_out.parameters():
            p.requires_grad = True
    
    def print_trainable_params(self):
        trainable = [
            name for name, p in self.named_parameters() if p.requires_grad
        ]
        self.print(f"Trainable params ({len(trainable)}):")
        for n in trainable:
            self.print(f"  {n}")


def train(cfg, log_dir):
    #print('Configuration:\n', cfg)

    #if 'project' in cfg:
    #    wandb.init(project=cfg.project, name=cfg.name)
    #else:
    #    cfg.name = 'test'

    # load the specified dataset
    if len(cfg.datasets) == 1: # one dataset training
        dataset = cfg.datasets[0]
        if dataset == 'fireprot':
            train_dataset = FireProtDataset(cfg, "train")
            val_dataset = FireProtDataset(cfg, "val")
        elif dataset == 'megascale_s669':
            train_dataset = MegaScaleDataset(cfg, "train_s669")
            val_dataset = MegaScaleDataset(cfg, "val")
        elif dataset.startswith('megascale_cv'):
                cv = dataset[-1]
                train_dataset = MegaScaleDataset(cfg, f"cv_train_{cv}")
                val_dataset = MegaScaleDataset(cfg, f"cv_val_{cv}")
        elif dataset == 'megascale':
                train_dataset = MegaScaleDataset(cfg, "train")
                val_dataset = MegaScaleDataset(cfg, "validation")
        else:
            raise ValueError("Invalid dataset specified!")
    else:
        train_dataset = ComboDataset(cfg, "train")
        val_dataset = ComboDataset(cfg, "val")

    if 'num_workers' in cfg.training:
        train_workers, val_workers = int(cfg.training.num_workers * 0.75), int(cfg.training.num_workers * 0.25)
    else:
        train_workers, val_workers = 0, 0

    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
    val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

    model_pl = TransferModelPL(cfg)
    model_pl.stage = 1

    tb_logger = TensorBoardLogger(save_dir=log_dir, name="", version="")

    cfg_name = getattr(cfg, "name", "experiment")
    filename = cfg_name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = 'val_ddG_spearman'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath=log_dir, filename=filename)


    #logger = WandbLogger(project=cfg.project, name="test", log_model="all") if 'project' in cfg else None
    max_ep = cfg.training.epochs if 'epochs' in cfg.training else 100

    #trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep, accelerator=cfg.platform.accel, devices=1)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=tb_logger, log_every_n_steps=10, max_epochs=max_ep, accelerator=cfg.platform.accel, devices=1)
    trainer.fit(model_pl, train_loader, val_loader)

    if 'two_stage' in cfg.training:  # sequential combo training
        if cfg.training.two_stage:
            print('Two-stage Training Enabled')
            del trainer, train_dataset, val_dataset, train_loader, val_loader
            # load new datasets for further training
            train_dataset = FireProtDataset(cfg, "train")
            val_dataset = MegaScaleDataset(cfg, "val")
            train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
            val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

            model_pl.stage = 2
            # re-start training with a new trainer
            trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep * 2,
                                accelerator=cfg.platform.accel, devices=1)
            trainer.fit(model_pl, train_loader, val_loader, ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments

    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("config", help="config file to use")

    # Create timestamp (e.g., 2025-09-22_14-30-05)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    # Build directory name
    log_dir = f"runs/{timestamp}"

    # Make sure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    args = parser.parse_args()
    yaml_file = args.config

    # Load config
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    cfg = OmegaConf.load(yaml_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("local.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    train(cfg, log_dir)
