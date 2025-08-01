import os
import ssl
import sys

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content, is_uv_available
from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    # First check: Ensure we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("\n" + "="*60)
        print("ERROR: Not running in a virtual environment!")
        print("Please activate your virtual environment first.")
        print("For example: source .venv/bin/activate")
        print("="*60 + "\n")
        sys.exit(1)
    
    # Check UV is available first
    if not is_uv_available():
        print("\n" + "="*60)
        print("ERROR: UV is required to run Fooocus.")
        print("Please install UV by running:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("="*60 + "\n")
        sys.exit(1)
    
    # Check for RTX 5090 environment variable
    use_rtx5090 = os.environ.get('USE_RTX5090', 'false').lower() == 'true'
    
    if use_rtx5090:
        torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/nightly/cu128")
        torch_command = os.environ.get('TORCH_COMMAND',
                                       f"uv pip install torch==2.9.0.dev20250726+cu128 torchvision==0.21.0.dev20250726+cu128 --index-url {torch_index_url}")
        requirements_file = os.environ.get('REQS_FILE', "requirements_rtx5090.txt")
    else:
        torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
        torch_command = os.environ.get('TORCH_COMMAND',
                                       f"uv pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}")
        requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("\n" + "="*60)
        print("ERROR: Not running in a virtual environment!")
        print("Please activate your virtual environment first.")
        print("For example: source .venv/bin/activate")
        print("="*60 + "\n")
        sys.exit(1)
    
    # Skip all automatic dependency installation
    # Dependencies should be managed manually
    print("Running in venv:", sys.prefix)
    print("Skipping automatic dependency installation (managed manually for RTX 5090)")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache

os.environ["U2NET_HOME"] = config.path_inpaint

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.util import get_file_from_folder_list

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not args.always_download_new_model:
        if not os.path.isfile(get_file_from_folder_list(default_model, config.paths_checkpoints)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(get_file_from_folder_list(alternative_model_name, config.paths_checkpoints)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_checkpoints))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

from webui import *
