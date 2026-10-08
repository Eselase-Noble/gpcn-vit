"""Resumable fine-tuning trainer for Baseline 0 (Colab-aware, §13).

Design goals:
  * weighted cross-entropy with FOLD-LOCAL class weights (passed in by the runner);
  * checkpoint every epoch + keep the best-by-monitor checkpoint, so a dropped
    Colab session resumes from the last epoch with optimizer/scheduler/RNG state;
  * per-epoch metrics written to JSON immediately (never lose a run's history);
  * model selection on VALIDATION only; the monitor defaults to threshold-free
    patient-level ROC-AUC (patient is the primary unit, spec).

The trainer optimises and selects; it does not choose the final decision
threshold or score the test fold — the runner does that with the evaluation
module, keeping test strictly held out.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.aggregation import aggregate_to_patient
from ..evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 3e-5
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    grad_clip: Optional[float] = 1.0
    log_every: int = 50
    extra: Dict = field(default_factory=dict)
    # Model selection = LOWEST validation loss (mean class-weighted CE over val
    # images). Patient/image ROC-AUC/PR-AUC are logged per epoch for diagnostics
    # only, never for checkpoint selection (audit decision 2026-10-08).


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_weights: Optional[List[float]],
        cfg: TrainConfig,
        ckpt_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        weight = (
            torch.tensor(class_weights, dtype=torch.float32, device=device)
            if class_weights is not None
            else None
        )
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        # Validation loss uses the SAME class weights as training (consistent
        # objective) but reduction='none' so we can report the mean over val
        # IMAGES exactly: mean_i( w_{y_i} * CE_i ).
        self.val_criterion = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
            if cfg.use_scheduler
            else None
        )
        self.start_epoch = 0
        self.best_val_loss = np.inf   # selection = minimise val loss
        self.best_epoch = -1
        self.history: List[Dict] = []
        self._fingerprint: Optional[Dict] = None  # set in fit()

    # ---- checkpointing ----------------------------------------------------

    @property
    def last_ckpt(self) -> Path:
        return self.ckpt_dir / "last.pt"

    @property
    def best_ckpt(self) -> Path:
        return self.ckpt_dir / "best.pt"

    def _save(self, epoch: int, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "cfg": self.cfg.__dict__,
            "fingerprint": self._fingerprint,
        }
        torch.save(state, self.last_ckpt)
        if is_best:
            torch.save(state, self.best_ckpt)
        (self.ckpt_dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def maybe_resume(self) -> None:
        """Resume from last.pt if present AND it matches this run (Colab recovery).

        A fingerprint (train size + epochs) guards against resuming a checkpoint
        from a different run that happens to share the directory (e.g. a smoke
        run vs a full run) — on mismatch we start fresh instead of corrupting state.
        """
        if not self.last_ckpt.exists():
            return
        # weights_only=False: these are our own checkpoints and include numpy/torch
        # RNG state objects for resumability (PyTorch 2.6+ default is True).
        state = torch.load(self.last_ckpt, map_location=self.device, weights_only=False)

        fp = state.get("fingerprint")
        if fp is not None and self._fingerprint is not None and fp != self._fingerprint:
            logger.warning(
                "Checkpoint fingerprint %s != current run %s — ignoring stale "
                "checkpoint and starting fresh.", fp, self._fingerprint)
            return

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.best_val_loss = state.get("best_val_loss", np.inf)
        self.best_epoch = state.get("best_epoch", -1)
        self.start_epoch = state["epoch"] + 1
        # RNG state must be a CPU ByteTensor; map_location may have moved it to GPU.
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"].cpu().to(torch.uint8))
        if state.get("numpy_rng") is not None:
            np.random.set_state(state["numpy_rng"])
        hist = self.ckpt_dir / "history.json"
        if hist.exists():
            self.history = json.loads(hist.read_text())
        logger.info("Resumed from epoch %d (best val_loss=%.4f @ epoch %d).",
                    self.start_epoch, self.best_val_loss, self.best_epoch)

    # ---- train / predict --------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running = 0.0
        for i, batch in enumerate(loader):
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if self.cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            running += loss.item() * x.size(0)
            if i % self.cfg.log_every == 0:
                logger.info("epoch %d step %d loss %.4f", epoch, i, loss.item())
        return running / len(loader.dataset)

    @torch.no_grad()
    def predict_df(self, loader: DataLoader) -> pd.DataFrame:
        """Return per-image predictions with patient_id/label/prob(malignant)."""
        self.model.eval()
        ds = loader.dataset
        probs, indices = [], []
        for batch in loader:
            x = batch["image"].to(self.device, non_blocking=True)
            p = torch.softmax(self.model(x), dim=1)[:, 1]  # P(malignant)
            probs.append(p.cpu().numpy())
            indices.append(batch["index"].numpy())
        if not probs:  # empty loader (e.g. a degenerate fold) — return empty frame
            return pd.DataFrame(columns=["filename", "patient_id", "label_binary", "prob"])
        probs = np.concatenate(probs)
        indices = np.concatenate(indices)
        meta = ds.df.iloc[indices]
        return pd.DataFrame({
            "filename": meta["filename"].to_numpy() if "filename" in meta else indices,
            "patient_id": meta["patient_id"].to_numpy(),
            "label_binary": meta["label_binary"].to_numpy(),
            "prob": probs,
        })

    @torch.no_grad()
    def _evaluate_val(self, val_loader: DataLoader) -> Dict:
        """Compute val loss (selection signal) + image/patient metrics (diagnostics).

        val_loss = mean over val IMAGES of the class-weighted CE, matching the
        training objective's weighting. Returns nan loss for an empty loader.
        """
        self.model.eval()
        preds = self.predict_df(val_loader)
        if len(preds) == 0:
            return {"val_loss": float("nan"), "image": {}, "patient": {}}

        total, n = 0.0, 0
        for batch in val_loader:
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            per_sample = self.val_criterion(self.model(x), y)  # w_{y_i} * CE_i
            total += float(per_sample.sum().item())
            n += x.size(0)
        val_loss = total / n if n else float("nan")

        image_m = compute_metrics(preds["label_binary"].to_numpy(), preds["prob"].to_numpy())
        agg = aggregate_to_patient(preds)
        patient_m = compute_metrics(agg["label_binary"].to_numpy(), agg["prob"].to_numpy())
        return {"val_loss": val_loss, "image": image_m, "patient": patient_m}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            resume: bool = False) -> Dict:
        """Train, selecting the checkpoint with the LOWEST validation loss.

        resume=False (default) starts fresh for a clean, traceable run; pass
        resume=True to continue an interrupted session from this ckpt_dir.
        """
        self._fingerprint = {
            "n_train": len(train_loader.dataset),
            "epochs": self.cfg.epochs,
        }
        if resume:
            self.maybe_resume()
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, epoch)
            if self.scheduler:
                self.scheduler.step()
            val = self._evaluate_val(val_loader)
            val_loss = val["val_loss"]
            is_best = np.isfinite(val_loss) and val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = float(val_loss)
                self.best_epoch = epoch
            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_image": val["image"],
                "val_patient": val["patient"],
                "is_best": bool(is_best),
                "seconds": round(time.time() - t0, 1),
            }
            self.history.append(rec)
            self._save(epoch, is_best)
            vp = val["patient"].get("roc_auc", float("nan"))
            logger.info("epoch %d done: train_loss=%.4f val_loss=%.4f "
                        "[diag val_patient_auc=%.3f]%s", epoch, train_loss, val_loss,
                        vp, "  *BEST*" if is_best else "")
        return {"best_epoch": self.best_epoch, "best_val_loss": self.best_val_loss,
                "epochs_run": self.cfg.epochs, "history": self.history}

    def load_best(self) -> None:
        if self.best_ckpt.exists():
            state = torch.load(self.best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            logger.info("Loaded best checkpoint (epoch %d).", state["epoch"])
