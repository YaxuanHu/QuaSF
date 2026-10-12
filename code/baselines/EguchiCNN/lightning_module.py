"""PyTorch Lightning module for EguchiCNN training."""
import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import EguchiCNN


class EguchiCNNLightningModule(pl.LightningModule):
    def __init__(self, max_len=512, num_classes=10, dropout=0.1,
                 lr=1e-3, lr_step_epoch=60, lr_factor=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = EguchiCNN(
            max_len=max_len,
            num_classes=num_classes,
            dropout=dropout,
        )
        # CrossEntropyLoss with ignore_index=0 to ignore padding/linker,
        # or use reduction='none' with mask
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, batch):
        x = batch['input']       # [B, 1, L, L]
        labels = batch['labels']  # [B, L]
        mask = batch['mask']      # [B, L]

        # Clamp labels that exceed num_classes to avoid CrossEntropyLoss crash
        out_of_range = labels >= self.hparams.num_classes
        if out_of_range.any():
            labels = labels.clone()
            mask = mask.clone()
            mask[out_of_range] = False   # exclude these residues from loss
            labels[out_of_range] = 0     # set to valid value (won't affect loss due to mask)

        logits = self.model(x)    # [B, C, L]

        # CrossEntropyLoss expects [B, C, L] and [B, L]
        loss_per_residue = self.loss_fn(logits, labels)  # [B, L]

        # Mask out padded positions
        loss_per_residue = loss_per_residue * mask.float()
        
        # Average over valid residues
        n_valid = mask.sum().clamp(min=1)
        loss = loss_per_residue.sum() / n_valid

        # Compute accuracy on valid residues
        preds = logits.argmax(dim=1)  # [B, L]
        correct = ((preds == labels) & mask).sum()
        acc = correct.float() / n_valid

        return loss, acc

    def training_step(self, batch, batch_idx):
        if batch is None or batch['mask'].sum() == 0:
            # Return a zero loss that participates in DDP gradient sync
            dummy = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy
        loss, acc = self._compute_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None or batch['mask'].sum() == 0:
            return None
        loss, acc = self._compute_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Paper: Adam, lr=0.001 for first 60 epochs, 0.0001 for next 100
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_epoch,
            gamma=self.hparams.lr_factor,
        )
        return [optimizer], [scheduler]
