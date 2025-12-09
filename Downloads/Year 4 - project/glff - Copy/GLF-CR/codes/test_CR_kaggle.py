import os
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import json

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from metrics import PSNR, SSIM
from dataloader import AlignedDataset, get_train_val_test_filelists
from net_CR_RDN import RDN_residual_CR

##########################################################
def test(CR_net, opts):
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    data = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True
    )

    # Report test set size
    test_set_size = len(test_filelist)
    print(f"Test set size: {test_set_size} images")

    total_psnr = 0.0
    total_ssim = 0.0
    results_per_image = []
    processed_images = 0

    iterator = tqdm(dataloader, total=len(dataloader), desc='Testing') if tqdm is not None else dataloader

    with torch.no_grad():
        for inputs in iterator:
            cloudy_data = inputs['cloudy_data'].cuda()
            cloudfree_data = inputs['cloudfree_data'].cuda()
            SAR_data = inputs['SAR_data'].cuda()
            file_names = inputs['file_name']

            pred_cloudfree_data = CR_net(cloudy_data, SAR_data)

            # compute numeric values robustly (support tensor or scalar)
            psnr_val = PSNR(pred_cloudfree_data, cloudfree_data)
            if hasattr(psnr_val, 'item'):
                psnr_val = float(psnr_val.item())
            else:
                psnr_val = float(psnr_val)

            ssim_val = SSIM(pred_cloudfree_data, cloudfree_data)
            if hasattr(ssim_val, 'item'):
                ssim_val = float(ssim_val.item())
            else:
                ssim_val = float(ssim_val)

            total_psnr += psnr_val * len(file_names)
            total_ssim += ssim_val * len(file_names)

            # Store per-image results (keeps compatibility with result saving)
            for file_name in file_names:
                results_per_image.append({
                    'image': file_name,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                })

            processed_images += len(file_names)

            # update progress bar postfix
            if tqdm is not None:
                iterator.set_postfix({'PSNR': f"{psnr_val:.3f}", 'SSIM': f"{ssim_val:.3f}", 'Done': processed_images})

            if getattr(opts, 'single_batch', False):
                break

    avg_psnr = total_psnr / processed_images if processed_images > 0 else 0.0
    avg_ssim = total_ssim / processed_images if processed_images > 0 else 0.0

    return avg_psnr, avg_ssim, results_per_image
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_sz', type=int, default=4, help='batch size used for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter', help='Path to S1 and S2 data')
    parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv', help='Path to data.csv')
    parser.add_argument('--checkpoint_path', type=str, default='/kaggle/input/checkpoint3/pytorch/default/1/checkpoint_epoch_2.pth', help='Path to model checkpoint')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False) 
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    parser.add_argument('--model_name', type=str, default='baseline', help='Model variant name for tracking')
    parser.add_argument('--notes', type=str, default='', help='Additional notes about this run')
    parser.add_argument('--single_batch', action='store_true', help='Run only a single batch')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='GPU IDs for multi-GPU (comma-separated)')

    opts = parser.parse_args()

    # Prefer generated CSV in /kaggle/working if it exists. If not, fall back to input dataset CSV.
    working_csv = '/kaggle/working/data.csv'
    input_csv = os.path.join(opts.input_data_folder, 'data.csv')

    # If user passed a custom path (not equal to the default working path), respect it.
    # Otherwise choose the best available CSV.
    if opts.data_list_filepath == working_csv:
        if os.path.exists(working_csv):
            opts.data_list_filepath = working_csv
        elif os.path.exists(input_csv):
            print(f"Note: generated CSV not found; falling back to dataset CSV: {input_csv}")
            opts.data_list_filepath = input_csv
        else:
            raise FileNotFoundError(f"No data CSV found. Checked: {working_csv} and {input_csv}")
    else:
        # user supplied a custom path; verify it exists
        if not os.path.exists(opts.data_list_filepath):
            raise FileNotFoundError(f"Specified data_list_filepath not found: {opts.data_list_filepath}")

    # Set GPU devices for multi-GPU support
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    gpu_count = len(opts.gpu_ids.split(','))

    print(f"\n{'='*60}")
    print(f"Available GPUs: {gpu_count}")
    print(f"GPU IDs: {opts.gpu_ids}")
    print(f"{'='*60}\n")

    # Load model
    CR_net = RDN_residual_CR(opts.crop_size).cuda()
    
    print(f"Loading checkpoint: {opts.checkpoint_path}")
    checkpoint = torch.load(opts.checkpoint_path, map_location='cuda', weights_only=False)
    
    # Handle both new and legacy checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'network' in checkpoint:
        state_dict = checkpoint['network']
    else:
        state_dict = checkpoint
    
    CR_net.load_state_dict(state_dict, strict=False)
    
    # Multi-GPU support
    if gpu_count > 1:
        print(f"Using DataParallel with {gpu_count} GPUs for inference\n")
        CR_net = nn.DataParallel(CR_net)

    CR_net.eval()
    for _, param in CR_net.named_parameters():
        param.requires_grad = False

    print(f"{'='*60}")
    print(f"Testing Model: {opts.model_name}")
    print(f"Input Data: {opts.input_data_folder}")
    print(f"Data CSV: {opts.data_list_filepath}")
    print(f"Checkpoint: {opts.checkpoint_path}")
    print(f"{'='*60}\n")
    print(f"{'Image':40s} | {'PSNR':>10s} | {'SSIM':>8s}")
    print("-" * 65)
    
    avg_psnr, avg_ssim, results_per_image = test(CR_net, opts)
    
    print("-" * 65)
    print(f"\n{'='*60}")
    print(f"Average Results:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  Total Images: {len(results_per_image)}")
    print(f"{'='*60}\n")
    
    # Save results to Kaggle working directory
    results_dir = '/kaggle/working/results' if os.path.exists('/kaggle/working') else '../results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_entry = {
        'timestamp': timestamp,
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': opts.model_name,
        'avg_psnr': float(avg_psnr),
        'avg_ssim': float(avg_ssim),
        'num_images': len(results_per_image),
        'notes': opts.notes,
        'per_image_results': results_per_image,
        'checkpoint': opts.checkpoint_path,
        'batch_sz': opts.batch_sz,
        'gpu_count': gpu_count
    }
    
    # Save individual run result
    result_file = os.path.join(results_dir, f'result_{opts.model_name}_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(result_entry, f, indent=4)
    
    # Append to history file
    history_file = os.path.join(results_dir, 'results_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(result_entry)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Create/update summary CSV for easy viewing
    summary_file = os.path.join(results_dir, 'results_summary.csv')
    file_exists = os.path.exists(summary_file) and os.path.getsize(summary_file) > 0
    with open(summary_file, 'a') as f:
        if not file_exists:
            f.write('Timestamp,Model,PSNR,SSIM,Num_Images,GPU_Count,Notes\n')
        f.write(f'{timestamp},{opts.model_name},{avg_psnr:.4f},{avg_ssim:.4f},{len(results_per_image)},{gpu_count},"{opts.notes}"\n')
    
    print(f"Results saved to:")
    print(f"  - {result_file}")
    print(f"  - {history_file}")
    print(f"  - {summary_file}\n")

if __name__ == "__main__":
    main()
