import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import json
from datetime import datetime
import time
from tqdm import tqdm

# Enable cuDNN autotuner for optimal performance
torch.backends.cudnn.benchmark = True

from dataloader import *
from model_CR_net import *
from metrics import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=4, help='batch size used for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')

parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/image1', help='Path to S1 and S2 data')
parser.add_argument('--data_list_filepath', type=str, default='/kaggle/input/image2/data.csv', help='Path to data.csv')
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--is_test', type=bool, default=False, help='whether in test mode')

parser.add_argument('--optimizer', type=str, default='Adam', help='Adam optimizer')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay step')
parser.add_argument('--lr_start_epoch_decay', type=int, default=5, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=10, help='maximum training epochs')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint every N epochs')
parser.add_argument('--save_model_dir', type=str, default='/kaggle/working/checkpoints', help='checkpoint directory')

parser.add_argument('--resume_checkpoint', type=str, default=None, help='path to resume checkpoint')
parser.add_argument('--train_from_scratch', action='store_true', default=True, help='train from scratch (no pretrained weights)')

parser.add_argument('--experiment_name', type=str, default='kaggle_training', help='experiment name for logging')
parser.add_argument('--notes', type=str, default='', help='additional notes')

parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs for multi-GPU training (comma-separated)')

opts = parser.parse_args()

##===================================================##
##************** Training functions *****************##
##===================================================##
def validate(model, val_dataloader):
    """Validate model on validation set"""
    model.net_G.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # Handle empty validation set gracefully
        try:
            val_len = len(val_dataloader)
        except Exception:
            val_len = None

        if val_len == 0:
            print("Warning: validation set is empty. Skipping validation.")
            model.net_G.train()
            return 0.0, 0.0

        progress_bar = tqdm(val_dataloader, desc="Validating", unit="batch")
        for data in progress_bar:
            model.set_input(data)
            pred = model.forward()
            
            batch_psnr = PSNR(pred, model.cloudfree_data)
            batch_ssim = SSIM(pred, model.cloudfree_data)
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_batches += 1
            
            # Update progress bar with current metrics
            avg_psnr = total_psnr / num_batches
            avg_ssim = total_ssim / num_batches
            progress_bar.set_postfix({'PSNR': f'{avg_psnr:.2f}', 'SSIM': f'{avg_ssim:.4f}'})
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    model.net_G.train()
    return avg_psnr, avg_ssim

def save_checkpoint_fn(model, epoch, val_psnr, best_val_psnr, opts):
    """Save training checkpoint"""
    os.makedirs(opts.save_model_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.net_G.state_dict(),
        'optimizer_state_dict': model.optimizer_G.state_dict(),
        'lr_scheduler_state_dict': model.lr_scheduler.state_dict(),
        'val_psnr': val_psnr,
        'best_val_psnr': best_val_psnr,
        'opts': opts
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(opts.save_model_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    is_best = val_psnr > best_val_psnr
    if is_best:
        best_path = os.path.join(opts.save_model_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path} (PSNR: {val_psnr:.2f} dB)")
    
    return is_best

##===================================================##
##******************** Main *************************##
##===================================================##
if __name__ == '__main__':
    ##===================================================##
    ##*************** Print configuration ***************##
    ##===================================================##
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    for arg in vars(opts):
        print(f"{arg:.<30} {getattr(opts, arg)}")
    print("="*60 + "\n")

    ##===================================================##
    ##*** Set GPU devices for multi-GPU training *******##
    ##===================================================##
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    gpu_count = len(opts.gpu_ids.split(','))
    print(f"GPUs available: {gpu_count}")
    print(f"GPU IDs: {opts.gpu_ids}\n")

    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    seed_torch()

    # Load train/val/test filelists.
    def read_csv_rows(path):
        rows = []
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for r in reader:
                    if len(r) == 0:
                        continue
                    rows.append(r)
        except Exception:
            return []
        return rows

    train_filelist, val_filelist, test_filelist = [], [], []
    basename = os.path.basename(opts.data_list_filepath).lower()
    parent_dir = os.path.dirname(opts.data_list_filepath)

    # If user provided combined data.csv (with split_id first column), use existing parser
    if 'data.csv' in basename:
        train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    else:
        # If user provided train.csv, try to find sibling val.csv and test.csv
        if 'train' in basename:
            train_filelist = read_csv_rows(opts.data_list_filepath)
            val_path = os.path.join(parent_dir, 'val.csv')
            test_path = os.path.join(parent_dir, 'test.csv')
            if os.path.exists(val_path):
                val_filelist = read_csv_rows(val_path)
            if os.path.exists(test_path):
                test_filelist = read_csv_rows(test_path)
            # If siblings not present, try combined data.csv in same directory
            if len(val_filelist) == 0 or len(test_filelist) == 0:
                combined_path = os.path.join(parent_dir, 'data.csv')
                if os.path.exists(combined_path):
                    t, v, te = get_train_val_test_filelists(combined_path)
                    # If combined has entries, prefer them for missing splits
                    if len(val_filelist) == 0:
                        val_filelist = v
                    if len(test_filelist) == 0:
                        test_filelist = te
        else:
            # Unknown filename: attempt to parse as combined or read as full list
            train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
            if len(train_filelist) == 0 and len(val_filelist) == 0 and len(test_filelist) == 0:
                # fallback: read all rows as train set
                all_rows = read_csv_rows(opts.data_list_filepath)
                train_filelist = all_rows

    print(f"Training samples: {len(train_filelist)}")
    print(f"Validation samples: {len(val_filelist)}")

    # Training dataloader with optimizations
    train_data = AlignedDataset(opts, train_filelist)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opts.batch_sz,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True if opts.num_workers > 0 else False,
        prefetch_factor=2 if opts.num_workers > 0 else None
    )

    # Validation dataloader
    val_data = AlignedDataset(opts, val_filelist)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True if opts.num_workers > 0 else False,
        prefetch_factor=2 if opts.num_workers > 0 else None
    )

    ##===================================================##
    ##****************** Create model *******************##
    ##===================================================##
    model = ModelCRNet(opts)

    # Note: Model is already on correct GPU in ModelCRNet.__init__
    # No need for DataParallel wrapping here

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_psnr = 0.0

    if opts.resume_checkpoint and os.path.exists(opts.resume_checkpoint):
        print(f"\nResuming from checkpoint: {opts.resume_checkpoint}")
        checkpoint = torch.load(opts.resume_checkpoint, map_location='cuda')
        
        # Handle DataParallel wrapping
        state_dict = checkpoint['model_state_dict']
        if gpu_count > 1 and not isinstance(model.net_G, nn.DataParallel):
            # Add 'module.' prefix if loading into DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('module.'):
                    new_state_dict[f'module.{k}'] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        elif not gpu_count > 1 and isinstance(model.net_G, nn.DataParallel):
            # Remove 'module.' prefix if loading into non-DataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        model.net_G.load_state_dict(state_dict)
        model.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
        model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'best_val_psnr' in checkpoint:
            best_val_psnr = checkpoint['best_val_psnr']
        print(f"Resuming from epoch {start_epoch}, best val PSNR: {best_val_psnr:.2f} dB\n")
    elif not opts.train_from_scratch:
        print("Warning: No checkpoint found but train_from_scratch=False")

    ##===================================================##
    ##**************** Train the network ****************##
    ##===================================================##
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    # Initialize logging
    training_log = {
        'experiment_name': opts.experiment_name,
        'notes': opts.notes,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(opts),
        'epochs': []
    }

    total_steps = 0
    train_start_time = time.time()

    for epoch in range(start_epoch, opts.max_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{opts.max_epochs-1}")
        print(f"{'='*60}")
        
        model.net_G.train()
        
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        # Wrap dataloader with tqdm for progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch}", unit="batch")
        for batch_idx, data in enumerate(progress_bar):
            total_steps += 1
            
            model.set_input(data)
            batch_loss = model.optimize_parameters()
            
            epoch_loss += batch_loss
            num_batches += 1
            
            # Update progress bar with current metrics
            avg_loss = epoch_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Epoch statistics
        avg_train_loss = epoch_loss / num_batches
        
        # Calculate training PSNR on last batch only (for speed)
        with torch.no_grad():
            avg_train_psnr = PSNR(model.pred_Cloudfree_data, model.cloudfree_data)
        
        # Validation
        print(f"\nRunning validation...")
        val_psnr, val_ssim = validate(model, val_dataloader)
        
        # Update learning rate
        if epoch >= opts.lr_start_epoch_decay:
            model.lr_scheduler.step()
            current_lr = model.optimizer_G.param_groups[0]['lr']
            print(f"Learning rate updated: {current_lr:.6f}")
        
        # Check if best model
        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr
        
        # Save checkpoint at configured frequency
        if (epoch % opts.save_freq == 0) or (epoch == opts.max_epochs - 1):
            save_checkpoint_fn(model, epoch, val_psnr, best_val_psnr, opts)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        epoch_log = {
            'epoch': epoch,
            'train_loss': float(avg_train_loss),
            'train_psnr': float(avg_train_psnr),
            'val_psnr': float(val_psnr),
            'val_ssim': float(val_ssim),
            'best_val_psnr': float(best_val_psnr),
            'learning_rate': float(model.optimizer_G.param_groups[0]['lr']),
            'epoch_time': float(epoch_time),
            'is_best': is_best
        }
        training_log['epochs'].append(epoch_log)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train PSNR: {avg_train_psnr:.2f} dB")
        print(f"  Val PSNR:   {val_psnr:.2f} dB")
        print(f"  Val SSIM:   {val_ssim:.4f}")
        print(f"  Best Val PSNR: {best_val_psnr:.2f} dB {'(NEW!)' if is_best else ''}")
        print(f"  Epoch Time: {epoch_time/60:.1f} minutes")
        print(f"{'='*60}")
        
        # Save training log after each epoch
        log_dir = os.path.join(opts.save_model_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{opts.experiment_name}_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    # Training complete
    total_time = time.time() - train_start_time
    training_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_log['total_time_hours'] = total_time / 3600

    # Final log save
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Training log saved to: {log_path}")
    print(f"Best model saved to: {os.path.join(opts.save_model_dir, 'best_model.pth')}")
    print(f"{'='*60}\n")
