import os
import json
from collections import defaultdict

import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

def load_emu2_visual_model(
    emu2_model_name_or_path: str = "BAAI/Emu2",
    emu2_checkpoint_dir: str = "/home/jongmin/.cache/huggingface/hub/models--BAAI--Emu2/snapshots/fa835ec101e52da5e081695107e1ddd3c7c4d88a",
    visual_layer_prefix: str = "model.visual.",
    target_visual_module_name: str = "visual",
    device: str = "cuda"
):
    """
    Emu2 모델의 visual backbone 부분을 로드합니다.

    Args:
        emu2_model_name_or_path (str): Hugging Face 모델명 또는 로컬 경로.
        emu2_checkpoint_dir (str): Emu2 모델 체크포인트(.bin 파일 및 index.json)가 있는 디렉토리 경로.
        visual_layer_prefix (str): 체크포인트 내 visual layer 가중치 키의 접두사.
        target_visual_module_name (str): image_model 내에서 실제 visual 가중치가 로드될 모듈의 이름.
        device (str): 모델을 로드할 디바이스 (e.g., "cuda", "cpu").

    Returns:
        torch.nn.Module: 로드된 Emu2 모델 (전체 모델 또는 visual 부분만일 수 있음, 현재는 전체 모델 반환 후 외부에서 처리 가정).
    """
    print(f"Initializing Emu2 model ({emu2_model_name_or_path}) with empty weights...")
    with init_empty_weights():
        image_model = AutoModelForCausalLM.from_pretrained(
            emu2_model_name_or_path,
            torch_dtype=torch.bfloat16, # 또는 필요에 따라 다른 dtype
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    # image_model.to_empty(device=device) # 수정된 코드: 모델을 먼저 디바이스로 옮김 (메타 텐서용)

    index_file_path = os.path.join(emu2_checkpoint_dir, "pytorch_model.bin.index.json")
    print(f"Reading checkpoint index file: {index_file_path}")
    with open(index_file_path, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map", {})

    shards_to_load_for_visual = defaultdict(list)
    visual_tensor_keys_in_checkpoint = []

    for tensor_name, shard_file_name in weight_map.items():
        if tensor_name.startswith(visual_layer_prefix):
            shards_to_load_for_visual[shard_file_name].append(tensor_name)
            visual_tensor_keys_in_checkpoint.append(tensor_name)

    if not visual_tensor_keys_in_checkpoint:
        print(f"Warning: No visual tensors found with prefix '{visual_layer_prefix}' in the checkpoint directory '{emu2_checkpoint_dir}'. Check the prefix and checkpoint files.")
        # 이 경우, image_model은 초기화된 상태로 반환될 수 있으나, 가중치는 로드되지 않음
        # 또는 에러를 발생시킬 수 있습니다. 여기서는 경고 후 진행합니다.

    print(f"Found {len(visual_tensor_keys_in_checkpoint)} visual tensors across {len(shards_to_load_for_visual)} shard(s) to load.")

    ## ===================================================================
    ## 텐서 데이터를 저장하기 위한 딕셔너리 생성
    ## ===================================================================
    visual_state_dict_for_load = {}
    loaded_tensor_count = 0
    for shard_file_name, tensors_in_this_shard_for_visual in shards_to_load_for_visual.items():
        shard_path = os.path.join(emu2_checkpoint_dir, shard_file_name)

        # 샤드 파일에 저장된 모든 텐서들을 CPU 메모리로 불러온다.
        # shard_state_dict는 샤드 파일 내의 모든 텐서를 담고 있는 딕셔너리가 된다. 
        # 이때, 시각적 부분과 관련 없는 텐서들도 포함될 수 있다.
        shard_state_dict = torch.load(shard_path, map_location="cpu")

        for full_tensor_name in tensors_in_this_shard_for_visual:
            # 텐서 이름에서 접두사 제거(e.g., model.visual.conv1.weight -> conv1.weight)
            # image_model의 실제 시각적 모듈에 가중치를 로드할 때는 접두사가 없는 키를 사용해야 한다.
            key_in_visual_module = full_tensor_name[len(visual_layer_prefix):]
            
            # 샤드에서 읽어온 실제 텐서 데이터(해당 모델 레이어에 대한 pretrained weight)를 가져와 저장한다.
            visual_state_dict_for_load[key_in_visual_module] = shard_state_dict[full_tensor_name].to(device)
            loaded_tensor_count += 1

        # 메모리 절약을 위해 shard_state_dict 삭제하고 CUDA 캐시를 비운다.
        del shard_state_dict
        torch.cuda.empty_cache()

    print(f"Prepared state_dict with {loaded_tensor_count} visual tensors for loading into the model on device '{device}'.")

    ## ===================================================================
    ## 이미지 모델에 pretrained weight 로드
    ## ===================================================================
    target_parent_module = None
    if hasattr(image_model, target_visual_module_name):
        target_parent_module = image_model
    elif hasattr(image_model, 'model') and hasattr(image_model.model, target_visual_module_name):
        # 일반적인 Hugging Face 모델 구조 (e.g., AutoModelForCausalLM의 경우 model.visual)
        target_parent_module = image_model.model

    target_module = getattr(target_parent_module, target_visual_module_name) # 가중치가 로드 될 대상 모듈
    print(f"Loading weights into target module (image_model...{target_visual_module_name})...")

    # 이미지 모델에 pretrained weight 로드
    missing_keys, unexpected_keys = target_module.load_state_dict(visual_state_dict_for_load, strict=True, assign=True)
    print(f"Successfully loaded weights into target module!")
    
    # 추출된 visual encoder를 eval 모드로 설정
    target_module.eval()
    print(f"Emu2 visual encoder ({target_visual_module_name}) loaded and set to eval mode.")

    # 추출된 visual encoder를 새로운 변수에 할당
    extracted_visual_module = target_module

    # 더 이상 필요 없는 원래의 전체 모델 삭제 (메모리 확보)
    # 먼저, target_module이 image_model의 직접적인 자식이 아닐 경우 (즉, image_model.model.visual 인 경우)
    # image_model.model 에서 visual을 None으로 설정하거나 삭제할 수 있습니다.
    # 하지만 가장 간단한 방법은 전체 image_model을 삭제하는 것입니다.
    # 주의: extracted_visual_module은 독립적인 객체가 아니라 image_model의 일부를 참조하고 있을 수 있습니다.
    # 파이썬의 가비지 컬렉션이 잘 동작하려면, image_model을 삭제하기 전에
    # extracted_visual_module이 image_model의 다른 부분에 대한 참조를 갖지 않도록 해야 합니다.
    # 이 경우, target_module은 이미 image_model의 일부이므로, image_model을 del 해도
    # extracted_visual_module은 계속 유효해야 하지만, 주의가 필요합니다.
    # 더 안전한 방법은 target_module을 깊은 복사하는 것이나, 이는 추가 메모리를 사용합니다.
    # 현재 구조에서는 target_module이 image_model의 하위 모듈이므로, image_model을 삭제하면
    # target_module에 대한 참조도 문제가 될 수 있습니다.
    # 따라서, image_model에서 visual 파트만 남기고 나머지를 제거하는 것이 더 안전할 수 있습니다.

    # 대안: image_model에서 visual 파트만 남기고 나머지를 제거 (더 복잡할 수 있음)
    # 예시:
    # for name, module in list(image_model.model.named_children()):
    # if name != target_visual_module_name:
    # delattr(image_model.model, name)
    # for name, param in list(image_model.model.named_parameters()):
    # if not name.startswith(target_visual_module_name):
    # del param # 이 방식은 파라미터 직접 삭제가 어려울 수 있음

    # 가장 간단하고 현재 구조에 맞는 접근:
    # visual_encoder를 반환하고, image_model 전체에 대한 참조는 audio_emu2.py에서 관리하도록 합니다.
    # load_emu2_visual_model 함수는 visual_encoder 모듈만 반환하는 책임을 집니다.
    # 호출한 쪽(audio_emu2.py)에서 전체 image_model이 더 이상 필요 없다면 그쪽에서 del image_model을 수행할 수 있습니다.

    # 현재 함수에서는 추출된 모듈만 반환하고, 전체 모델에 대한 참조는 남겨둡니다.
    # 메모리 정리는 호출하는 쪽에서 수행하는 것이 더 명확할 수 있습니다.
    # 만약 이 함수 내에서 전체 모델을 정리하고 싶다면,
    # extracted_visual_module = target_module
    # image_model = None # 또는 del image_model (호출 스코프에 영향 없을 시)
    # torch.cuda.empty_cache()
    # return extracted_visual_module

    # 현재로서는 target_module을 직접 반환합니다.
    # 호출부에서 Emu2_visual_model = load_emu2_visual_model(...) 로 받고,
    # 이 Emu2_visual_model이 실제로는 visual encoder 모듈 자체가 됩니다.
    
    # image_model.eval() # 전체 모델을 평가 모드로 설정 <-- 이 줄은 target_module.eval() 로 대체됨
    # print("Emu2 visual model parts loaded and model set to eval mode.") <-- 메시지 변경됨
    # return image_model <-- 이 줄이 extracted_visual_module (즉, target_module)을 반환하도록 변경됨
    return extracted_visual_module

if __name__ == '__main__':
    # 테스트를 위한 간단한 예시
    # 실제 사용 시에는 audio_emu2.py에서 이 함수를 호출합니다.
    # 경로 및 설정값은 실제 환경에 맞게 조정해야 합니다.
    # 이 테스트는 Emu2/ 디렉토리에서 실행한다고 가정합니다.

    # 기본값 사용 예시
    print("Testing Emu2 visual model loading with default parameters...")
    try:
        model = load_emu2_visual_model()
        print("Emu2 visual model loaded successfully for testing.")
        # print(model.visual) # 로드된 visual 모듈 확인
    except Exception as e:
        print(f"An error occurred during Emu2 visual model test loading: {e}")

    # 특정 체크포인트 경로 및 설정을 사용하는 예시 (필요시 주석 해제)
    # print("\nTesting Emu2 visual model loading with custom parameters...")
    # custom_ckpt_dir = "/path/to/your/emu2/snapshot"
    # if os.path.exists(custom_ckpt_dir):
    #     try:
    #         model_custom = load_emu2_visual_model(emu2_checkpoint_dir=custom_ckpt_dir)
    #         print("Emu2 visual model loaded successfully with custom checkpoint directory.")
    #     except Exception as e:
    #         print(f"Error with custom checkpoint: {e}")
    # else:
    #     print(f"Custom checkpoint directory {custom_ckpt_dir} not found. Skipping custom test.")