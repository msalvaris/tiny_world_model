"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from world_model import dataset
from world_model import utils


class Trainer:

    @staticmethod
    def get_default_config():
        C = utils.CfgNode()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.num_epochs = 10
        return C

    def __init__(self, config, model, train_dataset, validation_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @torch.no_grad()
    def run_validation(self):
        gpt_model, config = self.model, self.config
        # setup the dataloader
        validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        loss_accumulation = []
        progress_bar = tqdm(range(len(validation_loader)), leave=False)
        for idx, (batch, target) in enumerate(validation_loader):
            batch = batch.to(self.device)
            target = target.to(self.device)
            logits, self.val_loss = gpt_model(batch, targets=target.type(torch.long))

            progress_bar.set_postfix(
                validation_loss=self.val_loss.item()
            )  # report batch loss
            progress_bar.update(1)

            loss_accumulation.append(self.val_loss.item())
        progress_bar.close()
        return np.mean(loss_accumulation)

    def run(self):
        gpt_model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = gpt_model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        gpt_model.train()

        num_training_steps = config.num_epochs * len(train_loader)

        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(config.num_epochs):
            progress_bar.set_description(f"Epoch {epoch}")
            for idx, (batch, target) in enumerate(train_loader):
                batch = batch.to(self.device)
                target = target.to(self.device)
                logits, self.loss = gpt_model(batch, targets=target.type(torch.long))
                self.loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                progress_bar.set_postfix(loss=self.loss.item())  # report batch loss
                progress_bar.update(1)

                self.trigger_callbacks("on_batch_end")

            self.trigger_callbacks("on_epoch_end")

            if self.validation_dataset:
                gpt_model.eval()
                loss = self.run_validation()
                print(f"Epoch {epoch} validation loss {loss}")
                gpt_model.train()

        progress_bar.close()
