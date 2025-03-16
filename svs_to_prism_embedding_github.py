import os
import logging
import time
import cv2
import numpy as np
import torch
import timm
from tqdm import tqdm
import openslide
from PIL import Image
from transformers import AutoModel
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from multiprocessing import Pool, cpu_count
from huggingface_hub import login
import gc  # ADDED: import garbage collection

# -------- CONFIGURATION --------

# Replace with your actual token
huggingface_token = ""

# Login to Hugging Face
login(huggingface_token)

DATA_DIRS = {
    "Bladder": "/data/pa_cpgarchive/archives/bladder/TCGA_diagnostics/images/",  # Directory containing all .svs files
    "Breast": "/data/pa_cpgarchive/archives/breast/TCGA_diagnostics/images/original/",    # Directory containing subdirectories with .svs files
    "Prostate": "/data/pa_cpgarchive/archives/prostate/tcga/images/Diagnostic_image/"     # Directory containing subdirectories with .svs files
}

TILE_SIZE = 392
CROP_SIZE = 224
TISSUE_THRESHOLD = 0.65
NUM_WORKERS = max(1, cpu_count()//4)
BATCH_SIZE = 128
USE_GPU = torch.cuda.is_available()

device = torch.device("cuda" if USE_GPU else "cpu")

# -------- SETUP LOGGING --------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("wsi_processing.log")],
)

# -------- LOAD MODELS --------
logging.info(f"🖥️ Using {'GPU' if USE_GPU else 'CPU'} for inference.")

logging.info("🔄 Loading Virchow2 Model...")
virchow2_model = timm.create_model(
    "hf-hub:paige-ai/Virchow2",
    pretrained=True,
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU
)
virchow2_model.to(device).eval()
virchow2_transforms = create_transform(
    **resolve_data_config(virchow2_model.pretrained_cfg, model=virchow2_model)
)
logging.info("✅ Virchow2 Model Loaded Successfully")

logging.info("🔄 Loading PRISM Model...")
try:
    prism_model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
    prism_model.to(device).eval()
    logging.info("✅ PRISM Model Loaded Successfully")
except Exception as e:
    logging.error(f"❌ Failed to load PRISM model: {str(e)}")
    raise e

# -------- FILE SEARCH FUNCTIONS --------
def get_svs_files(directory, structured=True):
    """
    Finds all .svs files in a directory. 
    If `structured=True`, we assume each .svs is in its own subdirectory.
    """
    svs_files = []
    if structured:
        # For directories like Breast & Prostate
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith(".svs"):
                        svs_files.append(os.path.join(subdir_path, file))
    else:
        # For directories like Bladder
        svs_files = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(".svs")
        ]
    return svs_files

# -------- TILING & SEGMENTATION --------
def segment_tissue_downsampled(image_np, downsample_factor=4):
    """
    Quickly estimate tissue content in a tile by:
      1) Downsampling the tile
      2) Otsu thresholding on grayscale
    Returns a ratio (0 to 1) indicating fraction of the tile that is 'tissue'.
    """
    # Downsample
    h, w, _ = image_np.shape
    small = cv2.resize(image_np, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale and Otsu threshold
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tissue fraction in the downsampled image
    tissue_ratio_small = np.sum(binary > 0) / float(binary.size)
    return tissue_ratio_small

def chunked(iterable, chunk_size):
    """
    Yield successive chunks of size `chunk_size` from `iterable`.
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]

def process_tile_batch(args):
    """
    For a given batch of coordinates, open the slide once,
    read each tile, and return only those that meet the tissue threshold.
    """
    svs_path, coords_chunk, tile_size, tissue_threshold = args
    try:
        slide = openslide.OpenSlide(svs_path)
        valid_tiles = []
        for x, y in coords_chunk:
            try:   
                tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
                tile_np = np.array(tile)

                # Check tissue fraction with downsampled Otsu
                tissue_ratio = segment_tissue_downsampled(tile_np, downsample_factor=4)
                if tissue_ratio >= tissue_threshold:
                    valid_tiles.append(tile)
            except openslide.OpenSlideError as e:
                logging.error(f"Skipping tile at ({x}, {y}) from {svs_path}: {str(e)}")
        slide.close()
    except Exception as e:
        logging.error(f"Worker failed on {svs_path}: {e}", exc_info=True)
    return valid_tiles

def extract_tiles_parallel(svs_path):
    """
    Use multiprocessing in a chunked manner to reduce re-opening the slide.
    """
    # Open once here to get dimensions
    slide = openslide.OpenSlide(svs_path)
    w, h = slide.level_dimensions[0]
    slide.close()

    # Prepare all tile coordinates
    coords = []
    x_positions = list(range(0, w, TILE_SIZE))
    y_positions = list(range(0, h, TILE_SIZE))
    for x in x_positions:
        for y in y_positions:
            coords.append((x, y))

    # Chunk coordinates to reduce overhead
    chunk_size = max(1, len(coords) // (NUM_WORKERS * 2))  # heuristic
    tasks = [
        (svs_path, chunk, TILE_SIZE, TISSUE_THRESHOLD)
        for chunk in chunked(coords, chunk_size)
    ]

    all_tiles = []
    logging.info("⏳ Extracting tiles (multiprocessing, chunked) ...")
    with Pool(processes=NUM_WORKERS) as pool:
        for valid_tile_list in tqdm(pool.imap_unordered(process_tile_batch, tasks), total=len(tasks)):
            all_tiles.extend(valid_tile_list)

    logging.info(f"✅ Extracted {len(all_tiles)} tissue-rich tiles.")
    return all_tiles

# -------- MODEL INFERENCE --------
def process_tiles_with_virchow2(tiles):
    """
    Extract Virchow2 embeddings from a list of PIL tiles.
    Uses autocast (mixed precision) for speed if on GPU.
    """
    virchow2_embeddings = []

    # Process in mini-batches
    for i in tqdm(range(0, len(tiles), BATCH_SIZE), desc="Virchow2 embeddings"):
        batch_tiles = []
        for tile in tiles[i : i + BATCH_SIZE]:
            # Apply timm transforms
            tensor = virchow2_transforms(tile).unsqueeze(0)
            batch_tiles.append(tensor)

        batch_tensor = torch.cat(batch_tiles, dim=0).to(device)

        # Mixed precision if GPU
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_GPU), torch.no_grad():
            output = virchow2_model(batch_tensor)
            # output shape: [B, token_seq, hidden_dim]
            # Typically for a ViT-like, output[:, 0] is the CLS token
            # If the model has patch tokens, you can combine them as you like
            class_tokens = output[:, 0]       # [B, hidden_dim]
            patch_tokens = output[:, 5:]      # e.g., skip first 5 tokens if needed
            embeddings = torch.cat([class_tokens, patch_tokens.mean(dim=1)], dim=-1)
            virchow2_embeddings.append(embeddings)

    virchow2_embeddings = torch.cat(virchow2_embeddings, dim=0)
    return virchow2_embeddings

def process_with_prism(tile_embeddings):
    """
    Pass Virchow2 tile embeddings through PRISM,
    producing a slide-level embedding.
    """
    # If 2D, make it 3D for a single "slide" dimension
    if tile_embeddings.dim() == 2:
        tile_embeddings = tile_embeddings.unsqueeze(0)

    batch_size, tile_seq_len, _ = tile_embeddings.shape
    tile_mask = torch.ones((batch_size, tile_seq_len), dtype=torch.float16 if USE_GPU else torch.float32, device=device)

    # Convert embeddings to half if GPU is available
    if USE_GPU:
        tile_embeddings = tile_embeddings.half()

    # PRISM forward (with autocast float16)
    with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_GPU):
        reprs = prism_model.slide_representations(tile_embeddings, tile_mask=tile_mask)
        # reprs['image_embedding'] is the final slide-level embedding
        return reprs['image_embedding']

# -------- MAIN EXECUTION --------
def main():
    for category, directory in DATA_DIRS.items():
        logging.info(f"🚀 Processing {category} slides from {directory}")
        
        # Determine file structure
        structured = (category in ["Breast", "Prostate"])
        svs_files = get_svs_files(directory, structured=structured)

        # Create output dir
        output_dir = os.path.join("/data/temporary/amirhosein/", f"Prism_embedding_{category}")
        os.makedirs(output_dir, exist_ok=True)

        # Process each slide
        for svs_path in svs_files:
            slide_name = os.path.basename(svs_path)
            slide_root = os.path.splitext(slide_name)[0]  # strip the .svs extension

            # ADDED: Check if this slide's .pth already exists, skip if so
            existing_output_path = os.path.join(output_dir, f"{slide_root}.pth")
            if os.path.exists(existing_output_path):
                logging.info(f"✅ The file {existing_output_path} already exists. Skipping slide: {slide_name}")
                continue

            logging.info(f"🖼️ Processing slide: {slide_name}")

            # 1) Extract tissue-rich tiles (parallel)
            tiles = extract_tiles_parallel(svs_path)

            # 2) Virchow2 embeddings
            if len(tiles) == 0:
                logging.warning(f"No valid tiles found for {slide_name}. Skipping.")
                continue
            virchow2_embeddings = process_tiles_with_virchow2(tiles)

            # 3) PRISM: slide-level embedding
            prism_slide_embedding = process_with_prism(virchow2_embeddings)

            # 4) Save embedding
            save_path = os.path.join(output_dir, f"{slide_root}.pth")  # now just .pth
            torch.save(prism_slide_embedding.cpu(), save_path)
            logging.info(f"✅ Saved PRISM embedding: {save_path}")

            # ADDED: Force cleanup after processing this slide
            del tiles
            del virchow2_embeddings
            del prism_slide_embedding
            gc.collect()

    logging.info("🎉 All slides processed successfully!")

if __name__ == "__main__":
    main()
