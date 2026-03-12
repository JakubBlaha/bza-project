"""Download models from HuggingFace to a local cache directory."""

import logging
from pathlib import Path
from huggingface_hub import snapshot_download
from bza_tool.utils import setup_logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./hugging_cache")

def run_download(args) -> None:
    """CLI entry point for the ``download`` subcommand."""
    setup_logging()
    
    model_id = args.model_id
    # The local directory name should be the last part of the repo ID 
    # to match what's in the EasyEdit YAMLs (e.g. 'gpt2-xl')
    repo_name = model_id.split("/")[-1]
    
    target_dir = CACHE_DIR / repo_name
    
    logger.info("Downloading %s to %s...", model_id, target_dir)
    
    # ensure cache dir exists
    CACHE_DIR.mkdir(exist_ok=True)
    
    snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    
    logger.info("Download complete. Model available at: %s", target_dir)
