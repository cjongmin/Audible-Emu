import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
import numpy as np
import argparse
import os
import wandb
import importlib
import traceback
import librosa
from PIL import Image

from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torchvision.transforms as T
from transformers import AutoModelForCausalLM

from pretrained_model.CLAP import load_clap_model, get_clap_intermediate_patch_embeddings
from model.projector import Projector
from dataloaders.image_caption_dataset import ImageCaptionDataset

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", message=".*torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*")
warnings.filterwarnings("ignore", message=".*Some weights of RobertaModel were not initialized.*")
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


## ===================================================================
## Parse Arguments
## ===================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI602 Emu2 Audio Projector Training')
    
    g_data = parser.add_argument_group('Data settings')
    g_data.add_argument('--train_dataset_json', type=str, default='train10k.json', help='Path to the training dataset JSON file')
    g_data.add_argument('--audio_base_path', type=str, default='/mnt/lynx1/datasets/places205/', help='Base path for audio files specified in JSON')
    g_data.add_argument('--image_base_path', type=str, default='/mnt/lynx1/datasets/places205/vision/torralba/deeplearning/images256/', help='Base path for image files specified in JSON')
    g_data.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    g_data.add_argument('--skip_corrupted_data', action='store_true', help='If set, skips batches with corrupted data (e.g., unreadable images).')

    g_train = parser.add_argument_group('Training settings')
    g_train.add_argument('--max_epoch', type=int, default=100, help='Number of epochs for training')
    g_train.add_argument('--batch_size', type=int, default=50, help='Batch size for training')
    g_train.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the Projector')
    g_train.add_argument('--lossfunc', type=str, default="combined_loss", help='Loss function to use. Assumes a corresponding .py file in loss/ directory.')
    g_train.add_argument('--l2_weight', type=float, default=0.1, help='L2 weight for combined_loss (if applicable)')
    g_train.add_argument('--cosine_weight', type=float, default=1.0, help='Cosine weight for combined_loss (if applicable)')

    g_model = parser.add_argument_group('Model settings')
    g_model.add_argument('--clap_checkpoint_path', type=str, default='./music_speech_audioset_epoch_15_esc_89.98.pt', help='Path to CLAP model checkpoint')
    g_model.add_argument('--emu_model_type', type=str, default='emu2chat', choices=MODEL_CONFIGS.keys(), help=f'Type of Emu model to use: {list(MODEL_CONFIGS.keys())}')
    
    g_model.add_argument('--projector_input_patch_dim', type=int, default=512, help='Input patch dimension for Projector (e.g., CLAP intermediate layer output dim)')
    g_model.add_argument('--projector_input_num_patches', type=int, default=256, help='Number of input patches for Projector (e.g., CLAP intermediate layer sequence length)')
    
    g_model.add_argument('--projector_output_embed_dim', type=int, default=1792, help='Output embedding dimension for Projector (matching Emu visual token embed_dim)')
    g_model.add_argument('--projector_transformer_hidden_dim', type=int, default=768, help='Hidden dimension (d_model) for the Projector\'s Transformer. FFN dimension will be d_model * 4.')
    g_model.add_argument('--projector_num_transformer_layers', type=int, default=4, help='Total number of Transformer layers in the Projector. If 0, uses MLP. If 1, uses Encoder only. If >= 2, layers are split between Encoder and Decoder.')
    g_model.add_argument('--projector_num_heads', type=int, default=8, help='Number of attention heads in the Projector\'s Transformer')
    g_model.add_argument('--projector_dropout', type=float, default=0.1, help='Dropout rate for Projector Transformer')

    g_env = parser.add_argument_group('Environment settings')
    g_env.add_argument('--seed', type=int, default=2222, help='Random seed for reproducibility')
    g_env.add_argument('--save_path', type=str, default='./checkpoint/exp_projector_transformer', help='Location to save checkpoints')
    g_env.add_argument('--gpu', type=int, default=0, help='Primary GPU index to use')
    g_env.add_argument('--experiment_name', type=str, default='', help='Experiment name for WandB (auto-generated if empty)')
    g_env.add_argument('--wandb_project', type=str, default='emu2_audio_proj_transformer', help='WandB project name')

    return parser.parse_args()

def setup_environment(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu) 
        print(f"Primary GPU for Projector/CLAP set to: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    exp_name = args.experiment_name if args.experiment_name else f"proj_trans_e{args.max_epoch}_b{args.batch_size}_lr{args.lr}_loss_{args.lossfunc}"
    wandb.init(project=args.wandb_project, name=exp_name, config=args)
    
    print(f"Environment setup complete. Device: {device}. WandB run: {exp_name}")
    return device

## ===================================================================
## Define Model Configurations
## ===================================================================

MODEL_CONFIGS = {
    "emu2": {
        "name_or_path": "BAAI/Emu2",
        "snapshot_path": "/home/jongmin/.cache/huggingface/hub/models--BAAI--Emu2/snapshots/fa835ec101e52da5e081695107e1ddd3c7c4d88a",
        "n_query": 64,
        "default_image_size": 448
    },
    "emu2chat": {
        "name_or_path": "BAAI/Emu2-Chat",
        "snapshot_path": "/home/jongmin/.cache/huggingface/hub/models--BAAI--Emu2-Chat/snapshots/20ea30b04f8fee599cf97535e655c200df728501",
        "n_query": 256,
        "default_image_size": 448
    }
}

## ===================================================================
## Preprocessing Functions
## ===================================================================

def preprocess_audio_for_clap(audio_paths, target_sr=48000, target_duration_sec=10, device='cpu'):
    processed_waveforms = []
    target_length_samples = target_sr * target_duration_sec
    
    for audio_path in audio_paths:
        try:
            waveform, sr = librosa.load(audio_path, sr=target_sr)
            if len(waveform) < target_length_samples:
                padding = target_length_samples - len(waveform)
                waveform = np.pad(waveform, (0, padding), 'constant')
            elif len(waveform) > target_length_samples:
                waveform = waveform[:target_length_samples]
            processed_waveforms.append(torch.tensor(waveform, dtype=torch.float32))
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}. Skipping.")
            processed_waveforms.append(torch.zeros(target_length_samples, dtype=torch.float32))

    if not processed_waveforms:
        return None
    return torch.stack(processed_waveforms).to(device)

def preprocess_images_for_emu(pil_images, image_size, device='cpu'):
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
    ])
    processed_images = []
    for img_pil in pil_images:
        if img_pil is None:
             print(f"Warning: Encountered a None PIL image. Using placeholder.")
             processed_images.append(torch.zeros((3, image_size, image_size), dtype=torch.float32))
             continue
        try:
            processed_images.append(transform(img_pil))
        except Exception as e:
            print(f"Error transforming image: {e}. Using placeholder.")
            processed_images.append(torch.zeros((3, image_size, image_size), dtype=torch.float32))
    if not processed_images:
        return None
    return torch.stack(processed_images).to(device)

## ===================================================================
## Load and Prepare Models
## ===================================================================

def get_model_to_access(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def load_and_prepare_models(args, device):
    models = {}
    selected_emu_config = MODEL_CONFIGS[args.emu_model_type]
    emu_name_or_path = selected_emu_config['name_or_path']
    emu_snapshot_path = selected_emu_config['snapshot_path']
    actual_n_query_for_emu = selected_emu_config['n_query']

    print(f"Loading CLAP model from: {args.clap_checkpoint_path}...")
    clap_model = load_clap_model(checkpoint_path=args.clap_checkpoint_path, device=device)

    if hasattr(clap_model, 'model') and \
       hasattr(clap_model.model, 'audio_branch') and \
       hasattr(clap_model.model.audio_branch, 'spectrogram_extractor') and \
       hasattr(clap_model.model.audio_branch.spectrogram_extractor, 'stft'):
        try:
            clap_model.model.audio_branch.spectrogram_extractor.stft.pad_mode = 'constant'
            print("INFO: Changed STFT pad_mode to 'constant' in loaded CLAP model.")
        except Exception as e_pad:
            print(f"Warning: Could not change STFT pad_mode in loaded CLAP model: {e_pad}")
    else:
        print("Warning: Could not find STFT module in loaded CLAP model to change pad_mode.")

    models['clap'] = clap_model
    print(f"INFO: Projector input must match this function's output. Current Projector args: num_patches={args.projector_input_num_patches}, patch_dim={args.projector_input_patch_dim}")

    print(f"Loading Emu model structure from: {emu_name_or_path}...")
    with init_empty_weights():
        emu_full_model = AutoModelForCausalLM.from_pretrained(
            emu_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        max_memory_setting = {i: '20GiB' for i in range(num_gpus)}
        device_map = infer_auto_device_map(
            emu_full_model, 
            max_memory=max_memory_setting, 
            no_split_module_classes=['EmuBlock', 'EmuLayerNorm', 'LlamaDecoderLayer', 'Block']
        )
    elif torch.cuda.is_available():
        device_map = {"": device}
    else:
        device_map = {"": "cpu"}
        
    checkpoint_to_load_emu = emu_snapshot_path
    if not os.path.exists(emu_snapshot_path):
        print(f"Warning: Emu snapshot {emu_snapshot_path} not found. Loading from Hub: {emu_name_or_path}")
        checkpoint_to_load_emu = emu_name_or_path

    emu_full_model = load_checkpoint_and_dispatch(
        emu_full_model,
        checkpoint_to_load_emu,
        device_map=device_map,
    ).eval()

    emu_visual_device = device
    try:
        if hasattr(emu_full_model, 'model') and hasattr(emu_full_model.model, 'visual') and next(emu_full_model.model.visual.parameters(), None) is not None:
            emu_visual_device = next(emu_full_model.model.visual.parameters()).device
        else:
            for name, mod in emu_full_model.named_modules():
                if 'visual' in name.lower() and isinstance(mod, nn.Module) and list(mod.parameters(recurse=False)):
                    emu_visual_device = next(mod.parameters()).device
                    break
    except Exception as e:
        print(f"Warning: Error determining Emu visual device: {e}. Defaulting to {device}.")
    print(f"Emu visual encoder part expected on: {emu_visual_device}")

    print("Initializing Projector model...")
    projector_model = Projector(
        input_patch_dim=args.projector_input_patch_dim,
        num_input_patches=args.projector_input_num_patches,
        output_seq_len=actual_n_query_for_emu,
        output_embed_dim=args.projector_output_embed_dim,
        projector_transformer_hidden_dim=args.projector_transformer_hidden_dim,
        projector_num_transformer_layers=args.projector_num_transformer_layers,
        projector_num_heads=args.projector_num_heads,
        projector_dropout=args.projector_dropout
    ).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.gpu == 0:
        projector_model = nn.DataParallel(projector_model)
    
    models['emu_full'] = emu_full_model
    models['projector'] = projector_model
    models['emu_visual_device'] = emu_visual_device
    
    get_model_to_access(models['clap']).eval()
    for param in get_model_to_access(models['clap']).parameters():
        param.requires_grad = False
    
    for param in get_model_to_access(models['emu_full']).parameters():
        param.requires_grad = False

    get_model_to_access(models['projector']).train()

    if wandb.run is not None:
        wandb.watch(get_model_to_access(models['projector']), log='all', log_freq=100)
        
    return models

def custom_collate_fn(batch):
    pil_images = [item[0] for item in batch if isinstance(item[0], Image.Image)]
    other_data_list = [item[1:] if isinstance(item[0], Image.Image) else item for item in batch]
    collated_other_data = default_collate(other_data_list) if other_data_list and (len(other_data_list[0]) > 0 if other_data_list else False) else []

    if pil_images and not collated_other_data:
        return (pil_images,)
    elif not pil_images and collated_other_data:
        return tuple(collated_other_data) if isinstance(collated_other_data, list) else (collated_other_data,)
    elif pil_images and collated_other_data:
        return (pil_images, *collated_other_data) if isinstance(collated_other_data, list) else (pil_images, collated_other_data)
    return tuple()

def create_dataloader(args):
    train_dataset = ImageCaptionDataset(
        dataset_json_file=args.train_dataset_json,
        audio_base_path_override=args.audio_base_path,
        image_base_path_override=args.image_base_path,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True, # Drop last non-full batch
        collate_fn=custom_collate_fn # Use custom collate function
    )
    print(f"Train Dataloader created. Batches: {len(train_dataloader)}")
    return train_dataloader

def calculate_loss(audio_projected_embeds, image_encoded_embeds, args):
    try:
        loss_module_name = f'loss.{args.lossfunc}'
        loss_fn_class = importlib.import_module(loss_module_name).Loss 
        
        target_device = audio_projected_embeds.device
        
        image_encoded_embeds_for_loss = image_encoded_embeds.to(target_device)

        loss_val = loss_fn_class(
            audio_projected_embeds.float(), 
            image_encoded_embeds_for_loss.float().detach(),
            **vars(args)
        )
    except ModuleNotFoundError:
        raise ValueError(f"Loss module {loss_module_name}.py not found. Ensure it exists in loss/ directory.")
    except AttributeError:
        raise ValueError(f"Loss class 'Loss' not found in module {loss_module_name}.")
    except Exception as e:
        print(f"Error in loss calculation ({args.lossfunc}): {e}")
        traceback.print_exc()
        raise e
        
    if isinstance(loss_val, list):
        loss_val = torch.stack(loss_val).mean()
        
    return loss_val

def train_one_epoch(epoch, models, dataloader, optimizer, device, args):
    projector_model = models['projector']
    clap_model = models['clap'] 
    emu_full_model = models['emu_full']
    emu_visual_device = models['emu_visual_device']

    get_model_to_access(projector_model).train()
    total_loss, num_batches_processed = 0, 0
    selected_emu_config = MODEL_CONFIGS[args.emu_model_type]
    image_size_for_emu = selected_emu_config['default_image_size']

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.max_epoch}]")

    for batch_idx, batch_data in enumerate(progress_bar):
        if not batch_data:
            print(f"Skipping empty batch: epoch {epoch+1}, batch {batch_idx}")
            continue
            
        pil_images, audio_paths, _ = batch_data
        
        valid_indices = [i for i, img in enumerate(pil_images) if img is not None]
        if len(valid_indices) != len(pil_images):
            print(f"Warning: Batch {batch_idx} contains None images. Filtering them out.")
            if not valid_indices:
                print(f"Skipping batch {batch_idx}: all images are None.")
                continue
            pil_images = [pil_images[i] for i in valid_indices]
            audio_paths = [audio_paths[i] for i in valid_indices]
        
        if not pil_images:
            print(f"Skipping batch {batch_idx}: no valid images after filtering.")
            continue

        optimizer.zero_grad()

        try: # Audio Branch
            audio_waveforms_batch = preprocess_audio_for_clap(audio_paths, device=device)
            if audio_waveforms_batch is None or audio_waveforms_batch.shape[0] == 0:
                print(f"Skipping batch {batch_idx}: audio preprocessing failure.")
                continue

            with torch.no_grad():
                clap_intermediate_patches = get_clap_intermediate_patch_embeddings(
                    clap_model_instance=clap_model,
                    audio_waveforms=audio_waveforms_batch,
                    device=device
                )

            if clap_intermediate_patches.shape[1] != args.projector_input_num_patches or \
               clap_intermediate_patches.shape[2] != args.projector_input_patch_dim:
                print(f"ERROR: Mismatch CLAP patches shape {clap_intermediate_patches.shape} vs Projector input ({args.projector_input_num_patches},{args.projector_input_patch_dim}). Skipping batch {batch_idx}.")
                continue
            
            projected_audio_embeddings = projector_model(clap_intermediate_patches)

        except Exception as e:
            print(f"Error in Audio Branch (batch {batch_idx}): {e}")
            traceback.print_exc()
            continue

        try:
            image_pixels_batch = preprocess_images_for_emu(pil_images, image_size=image_size_for_emu, device=emu_visual_device)
            if image_pixels_batch is None or image_pixels_batch.shape[0] == 0:
                print(f"Skipping batch {batch_idx}: image preprocessing failure.")
                continue
            
            target_emu_dtype = get_model_to_access(emu_full_model).dtype
            if callable(target_emu_dtype):
                target_emu_dtype = target_emu_dtype()
            
            image_pixels_batch = image_pixels_batch.to(dtype=target_emu_dtype)

            emu_model_comp = get_model_to_access(emu_full_model).model
            if hasattr(emu_model_comp, 'encode_image') and callable(getattr(emu_model_comp, 'encode_image')):
                image_token_embeddings_emu = emu_model_comp.encode_image(image_pixels_batch)
            else:
                print(f"Error: Cannot find 'encode_image' in Emu model (batch {batch_idx}).")
                continue

        except Exception as e:
            print(f"Error in Image Branch (batch {batch_idx}): {e}")
            traceback.print_exc()
            continue

        try:
            loss = calculate_loss(projected_audio_embeddings, image_token_embeddings_emu, args)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches_processed += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / num_batches_processed:.4f}")

            if (batch_idx + 1) % 20 == 0 and wandb.run is not None:
                wandb.log({
                    "epoch": epoch + 1,
                    "batch_idx": batch_idx + 1,
                    "batch_loss": loss.item(),
                    "running_avg_loss_epoch": total_loss / num_batches_processed
                })
        except Exception as e:
            print(f"Error in Loss/Optimization (batch {batch_idx}): {e}")
            traceback.print_exc()
            continue

    return total_loss / num_batches_processed if num_batches_processed > 0 else 0

def main():
    args = parse_arguments()
    if torch.cuda.is_available():
        torch.cuda.init()
        
    device = setup_environment(args)
    models = load_and_prepare_models(args, device)
    optimizer = optim.Adam(get_model_to_access(models['projector']).parameters(), lr=args.lr)
    train_loader = create_dataloader(args)

    print("Starting training...")
    for epoch in range(args.max_epoch):
        avg_epoch_loss = train_one_epoch(epoch, models, train_loader, optimizer, device, args)
        print(f"Epoch [{epoch+1}/{args.max_epoch}] completed. Avg Loss: {avg_epoch_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"epoch_completed": epoch + 1, "average_epoch_loss": avg_epoch_loss})

        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            save_path = os.path.join(args.save_path, f"projector_epoch_{epoch+1:03d}.pt")
            torch.save(get_model_to_access(models['projector']).state_dict(), save_path)
            print(f"Projector model saved to {save_path}")

    print("Training finished.")
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
