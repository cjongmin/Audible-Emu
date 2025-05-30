import json
# import librosa # No longer used directly here for main audio loading
import numpy as np
import os
from PIL import Image
# import scipy # No longer used here
import torch
from torch.utils.data import Dataset
# import torchvision.transforms as transforms # No longer used here

# preemphasis function can be removed if not used elsewhere, or kept if general utility

class ImageCaptionDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf=None, image_conf=None, audio_base_path_override=None, image_base_path_override=None):
        """
        Dataset that manages a set of paired images and audio recordings.
        Returns PIL Image and audio file path.

        :param dataset_json_file: Path to the dataset JSON file.
        :param audio_conf: (Optional) Kept for compatibility, but largely unused as audio path is returned.
        :param image_conf: (Optional) Kept for compatibility, but largely unused as PIL image is returned.
        :param audio_base_path_override: (Optional) Override for audio base path.
        :param image_base_path_override: (Optional) Override for image base path.
        """
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        
        # Base paths can be set from defaults or overridden
        self.image_base_path = image_base_path_override if image_base_path_override is not None else '/mnt/lynx1/datasets/places205/vision/torralba/deeplearning/images256/'
        self.audio_base_path = audio_base_path_override if audio_base_path_override is not None else '/mnt/lynx1/datasets/places205/'

        # audio_conf and image_conf are stored but not heavily used for transformations within the dataset itself anymore
        self.audio_conf = audio_conf if audio_conf else {}
        self.image_conf = image_conf if image_conf else {}
        
        # The following transformation initializations are removed as per request
        # self.image_resize_and_crop = ...
        # self.image_normalize = ...
        # self.windows = ...

    # _LoadAudio and _LoadImage methods are removed as their logic is simplified into __getitem__
    # or will be handled externally by the training script's preprocessing functions.

    def __getitem__(self, index):
        """
        returns: (image_pil, audio_path, caption_text)
        image_pil: PIL.Image object (RGB)
        audio_path: string, path to the audio file
        caption_text: string, caption associated with the image/audio
        """
        datum = self.data[index]
        
        wavpath = os.path.join(self.audio_base_path, datum['wav'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        caption = datum.get('caption', "") # Get caption, default to empty string if not present

        try:
            image_pil = Image.open(imgpath).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {imgpath}. Returning None. Error: {e}")
            image_pil = None # Or a placeholder PIL image

        # audio_path is simply the path, actual loading and processing will happen in the training loop / dataloader collate_fn
        
        return image_pil, wavpath, caption # Return PIL image, audio file path, and caption

    def __len__(self):
        return len(self.data)

# Example of how one might handle None images in collate_fn if returned by __getitem__
def collate_fn_skip_corrupted(batch):
    # Filter out None items (e.g., if an image failed to load)
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None # Or raise an error, or return an empty batch structure
    return torch.utils.data.dataloader.default_collate(batch)
