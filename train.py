"""
Training script for Multi-Frame CRNN License Plate Recognition.

Usage:
    python train.py

The data directory should be configured in config.py (DATA_ROOT).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Support both running as module and direct script execution
try:
    from .config import Config
    from .dataset import AdvancedMultiFrameDataset
    from .models import MultiFrameCRNN
    from .utils import seed_everything, decode_predictions
except ImportError:
    from config import Config
    from dataset import AdvancedMultiFrameDataset
    from models import MultiFrameCRNN
    from utils import seed_everything, decode_predictions


def train_pipeline(pretrained_weights=None, start_epoch=0, checkpoint_path="checkpoint.pth", epochs=None):
    """Main training pipeline.

    Args:
        pretrained_weights: Path to pretrained model weights (.pth) for fine-tuning.
                            Chỉ load weights model, không restore optimizer/scheduler.
        start_epoch:        Epoch bắt đầu (0 = train từ đầu).
                            Nếu > 0, load full checkpoint từ checkpoint_path để resume.
        checkpoint_path:    Đường dẫn save/load checkpoint sau mỗi epoch.
        epochs:             Tổng số epochs cần train. Mặc định dùng Config.EPOCHS.
    """
    num_epochs = epochs if epochs is not None else Config.EPOCHS
    seed_everything(Config.SEED)
    print(f"🚀 TRAINING START | Device: {Config.DEVICE} | Epochs: {start_epoch + 1} → {num_epochs}")

    # Check data directory
    if not os.path.exists(Config.DATA_ROOT):
        print(f"❌ LỖI: Sai đường dẫn DATA_ROOT: {Config.DATA_ROOT}")
        return

    # Create datasets
    train_ds = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='train', split_ratio=0.8)
    val_ds   = AdvancedMultiFrameDataset(Config.DATA_ROOT, mode='val',   split_ratio=0.8)

    if len(train_ds) == 0:
        print("❌ Dataset Train rỗng!")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=AdvancedMultiFrameDataset.collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=AdvancedMultiFrameDataset.collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        print("⚠️  Validation set rỗng — bỏ qua validate.")
        val_loader = None

    # ── Model ────────────────────────────────────────────────────────────────
    model = MultiFrameCRNN(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)

    # Load pretrained weights (model only, dùng khi fine-tune lần đầu)
    if pretrained_weights and os.path.exists(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location=Config.DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded pretrained weights: {pretrained_weights}")
    elif pretrained_weights:
        print(f"⚠️  Weights không tìm thấy: {pretrained_weights} — train từ đầu")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"🖥️  DataParallel — sử dụng {num_gpus} GPU")
    else:
        print(f"🖥️  Device: {Config.DEVICE}")

    # raw_model dùng để access state_dict gốc khi đang wrap DataParallel
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # ── Optimizer / Scheduler / Scaler ───────────────────────────────────────
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
    )
    use_amp = Config.DEVICE.type == 'cuda'
    scaler  = GradScaler(enabled=use_amp)

    best_acc = 0.0

    # ── Resume từ checkpoint ──────────────────────────────────────────────────
    if start_epoch > 0:
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE)
            raw_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            best_acc = ckpt.get('best_acc', 0.0)
            print(f"▶️  Resumed checkpoint: epoch {ckpt['epoch'] + 1}, best_acc={best_acc:.2f}%")
        else:
            print(f"⚠️  Checkpoint không tìm thấy tại {checkpoint_path} — bắt đầu từ epoch {start_epoch + 1} với weights hiện tại")

    # ── Helper save checkpoint ────────────────────────────────────────────────
    def save_checkpoint(epoch):
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'best_acc':             best_acc,
        }, checkpoint_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch = start_epoch  # giữ scope cho except block
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0

            pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{num_epochs}")
            for images, targets, target_lengths, _ in pbar:
                images  = images.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with autocast(Config.DEVICE.type, enabled=use_amp):
                    preds = model(images)
                    preds_permuted = preds.permute(1, 0, 2)
                    input_lengths  = torch.full(
                        size=(images.size(0),),
                        fill_value=preds.size(1),
                        dtype=torch.long,
                    )
                    loss = criterion(preds_permuted, targets, input_lengths, target_lengths)

                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= scale_before:
                    scheduler.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

            avg_train_loss = epoch_loss / len(train_loader)

            # ── Validation ────────────────────────────────────────────────────
            val_acc      = 0.0
            avg_val_loss = 0.0

            if val_loader:
                model.eval()
                val_loss       = 0
                total_correct  = 0
                total_samples  = 0

                with torch.no_grad():
                    for images, targets, target_lengths, labels_text in val_loader:
                        images  = images.to(Config.DEVICE)
                        targets = targets.to(Config.DEVICE)
                        preds   = model(images)

                        loss = criterion(
                            preds.permute(1, 0, 2),
                            targets,
                            torch.full((images.size(0),), preds.size(1), dtype=torch.long),
                            target_lengths,
                        )
                        val_loss += loss.item()

                        decoded = decode_predictions(torch.argmax(preds, dim=2), Config.IDX2CHAR)
                        for i in range(len(labels_text)):
                            if decoded[i] == labels_text[i]:
                                total_correct += 1
                        total_samples += len(labels_text)

                avg_val_loss = val_loss / len(val_loader)
                val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

            print(f"Ep {epoch + 1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.2f}%")

            # ── Save checkpoint sau mỗi epoch ─────────────────────────────────
            save_checkpoint(epoch)

            # ── Save best model ───────────────────────────────────────────────
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(raw_model.state_dict(), "best_model.pth")
                print(f" -> ⭐ Saved Best Model! ({val_acc:.2f}%)")

    except KeyboardInterrupt:
        print(f"\n⚠️  Bị ngắt tại epoch {epoch + 1}! Đang lưu checkpoint...")
        save_checkpoint(epoch)
        print(f"✅ Checkpoint lưu tại: {checkpoint_path}")
        print(f"   Resume bằng cách đặt start_epoch={epoch + 1}")


if __name__ == "__main__":
    train_pipeline()
