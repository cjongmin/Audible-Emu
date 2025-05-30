import laion_clap
import torch

def load_clap_model(checkpoint_path: str, device: str = "cuda"):
    """
    CLAP 오디오 모델을 로드하고 지정된 디바이스로 옮깁니다.

    Args:
        checkpoint_path (str): CLAP 모델 체크포인트 파일 경로.
        device (str): 모델을 로드할 디바이스 (e.g., "cuda", "cpu").

    Returns:
        laion_clap.CLAP_Module: 로드된 CLAP 모델.
    """
    # amodel을 'HTSAT-base'로 고정하거나, 필요시 파라미터로 받을 수 있습니다.
    audio_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    audio_model.load_ckpt(checkpoint_path, verbose=False)
    audio_model = audio_model.to(device)
    audio_model.eval()
    print(f"CLAP audio model loaded from {checkpoint_path} and moved to {device}.")
    return audio_model

def get_clap_intermediate_patch_embeddings(clap_model_instance, audio_waveforms, device):
    audio_encoder = clap_model_instance.model.audio_branch
    audio_encoder.to(device)
    audio_encoder.eval()

    if audio_waveforms.ndim == 3 and audio_waveforms.shape[1] == 1:
        x_for_stft = audio_waveforms.squeeze(1)
    elif audio_waveforms.ndim == 2:
        x_for_stft = audio_waveforms
    else:
        raise ValueError(f"[Error] Invalid input shape: {audio_waveforms.shape}")

    x_for_stft = x_for_stft.to(device, dtype=torch.float32)

    x = audio_encoder.spectrogram_extractor(x_for_stft)
    x = audio_encoder.logmel_extractor(x)
    x = x.transpose(1, 3)
    x = audio_encoder.bn0(x)
    x = x.transpose(1, 3)

    x = audio_encoder.spec_augmenter(x)

    _, _, T_orig, F_orig = x.shape

    spec_s = audio_encoder.spec_size
    mel_b = getattr(audio_encoder.config, 'mel_bins', 64) 
    freq_r = spec_s // mel_b

    target_T_for_reshape = int(spec_s * freq_r)
    target_F_for_reshape = spec_s // freq_r
    
    if T_orig != target_T_for_reshape or F_orig != target_F_for_reshape:
        x = torch.nn.functional.interpolate(
            x, size=(target_T_for_reshape, target_F_for_reshape), mode="bicubic", align_corners=True
        )

    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], freq_r, x.shape[3] // freq_r)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
    x = audio_encoder.patch_embed(x)
    x = audio_encoder.pos_drop(x)
    x, _ = audio_encoder.layers[0](x)

    x, _ = audio_encoder.layers[1](x)

    layer = audio_encoder.layers[2]
    for _, blk in enumerate(layer.blocks):
        x, _ = blk(x)

    return x