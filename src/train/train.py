from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import random
import time
import numpy as np

from .ictone_dataset import ICToneDataset, ICTonePairDataset, ICToneTripletDataset
from .model import OminiModel
from .callbacks import TrainingCallback


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Optional LoRA warm-start override from env. Empty / unset -> keep the
    # yaml value (which defaults to null = from-scratch LoRA init).
    env_lora = os.environ.get("XFL_LORA_PATH")
    if env_lora:
        config.setdefault("train", {})["lora_path"] = env_lora
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    # Prefer the run name exported by the launch script (so train.log and
    # checkpoints/samples land in the same runs/<name> dir). Fall back to a
    # locally-generated timestamp if the env var is not set.
    run_name = os.environ.get("XFL_RUN_NAME") or time.strftime("%Y%m%d-%H%M%S")

    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    if 'use_offset_noise' not in config.keys():
        config['use_offset_noise'] = False

    # Initialize dataset and dataloader
    ds_type = training_config["dataset"]["type"]
    if ds_type == "ictone":
        # Filter-migration triplet dataset (content | style | GT), 512x1536
        # canvas with the right third masked out. See
        # ``src/train/ictone_dataset.py`` for triplet-construction details.
        ds_cfg = training_config["dataset"]
        dataset = ICToneDataset(
            source_dir=ds_cfg["source_dir"],
            lut_list_path=ds_cfg["lut_list_path"],
            condition_size=ds_cfg["condition_size"],
            target_size=ds_cfg["target_size"],
            drop_text_prob=ds_cfg.get("drop_text_prob", 0.1),
            instance_prompt=ds_cfg.get("instance_prompt"),
            length=ds_cfg.get("length"),
            repeats=ds_cfg.get("repeats", 1),
        )

    elif ds_type == "ictone_pair":
        # Pair-driven variant: sample rows from a merged sampled_pairs npz
        # and derive (q_image, r_image, q_lut, r_lut). Content branch keeps
        # the original random multi-LUT chain applied to q_image.
        ds_cfg = training_config["dataset"]
        dataset = ICTonePairDataset(
            pairs_npz=ds_cfg["pairs_npz"],
            image_names_txt=ds_cfg["image_names_txt"],
            lut_index_tsv=ds_cfg["lut_index_tsv"],
            content_lut_list_path=ds_cfg.get("content_lut_list_path"),
            condition_size=ds_cfg["condition_size"],
            target_size=ds_cfg["target_size"],
            drop_text_prob=ds_cfg.get("drop_text_prob", 0.1),
            instance_prompt=ds_cfg.get("instance_prompt"),
            length=ds_cfg.get("length"),
            repeats=ds_cfg.get("repeats", 1),
            content_k_min=ds_cfg.get("content_k_min", 2),
            content_k_max=ds_cfg.get("content_k_max", 4),
            cache_luts=ds_cfg.get("cache_luts", True),
        )

    elif ds_type == "ictone_triplet":
        # Triplet-driven variant: reads pre-built (content, reference, gt)
        # triples from a triplet.json file. Reference and GT are used
        # verbatim; the content branch still receives a random multi-LUT
        # degradation chain (K-1 LUTs, K in [content_k_min, content_k_max]).
        ds_cfg = training_config["dataset"]
        dataset = ICToneTripletDataset(
            triplet_json=ds_cfg["triplet_json"],
            data_root=ds_cfg.get("data_root", ""),
            content_lut_list_path=ds_cfg.get("content_lut_list_path"),
            lut_index_tsv=ds_cfg.get("lut_index_tsv"),
            condition_size=ds_cfg["condition_size"],
            target_size=ds_cfg["target_size"],
            drop_text_prob=ds_cfg.get("drop_text_prob", 0.1),
            instance_prompt=ds_cfg.get("instance_prompt"),
            length=ds_cfg.get("length"),
            repeats=ds_cfg.get("repeats", 1),
            content_k_min=ds_cfg.get("content_k_min", 2),
            content_k_max=ds_cfg.get("content_k_max", 4),
            cache_luts=ds_cfg.get("cache_luts", True),
        )

    else:
        raise ValueError(
            f"Unsupported dataset.type={ds_type!r}. ICTone supports "
            f"'ictone', 'ictone_pair', and 'ictone_triplet'."
        )

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_fill_id=config["flux_path"],
        lora_path=training_config.get("lora_path"),
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        use_offset_noise=config["use_offset_noise"],
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
