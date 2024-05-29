import os

import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import fire
from datetime import datetime
import sys
from rich.progress import Progress, SpinnerColumn,TextColumn, TaskProgressColumn, TimeRemainingColumn
import atexit

from world_model import dataset, model, train, utils, movies


def _prediction_generator(seq_path, root_folder, transform, gpt_model, sequence, device):
    gpt_model.eval()
    with torch.no_grad():
        for paths in generate_paths(seq_path):
            example = torch.stack(
                [
                    _convert_and_transform(
                        os.path.join(root_folder, sequence, ex), transform=transform
                    )
                    for ex in paths
                ]
            )
            # example needs to be shape b x t x (64*64), b=1,t=4
            batch = example.to(device)
            pred, _ = gpt_model(batch.unsqueeze(dim=0))
            pred_array = (
                torch.nn.functional.softmax(pred, dim=1)
                .argmax(dim=1)
                .detach()
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)
            yield pred_array.squeeze() # Remove batch dimension


def _ground_truth_generator(seq_path,root_folder, sequence):
    img_paths = sorted(os.listdir(seq_path), key=_custom_sorter)
    for path in img_paths[4:]: # Only loop through the images we will generate predictions for
        yield np.array(Image.open(os.path.join(root_folder, sequence, path)))


def _create_movie(
    seq_path, root_folder, transform, gpt_model, sequence, save_name, device
):
    
    dirname = os.path.dirname(save_name)
    os.makedirs(dirname, exist_ok=True)

    # Create a writer object
    writer = imageio.get_writer(
        save_name, fps=10
    )  # Adjust the fps as needed
    
    gpt_model.eval()
    with torch.no_grad():
        for idx, paths in enumerate(generate_paths(seq_path)):
            example = torch.stack(
                [
                    _convert_and_transform(
                        os.path.join(root_folder, sequence, ex), transform=transform
                    )
                    for ex in paths
                ]
            )
            # example needs to be shape b x t x (64*64), b=1,t=4
            batch = example.to(device)
            pred, _ = gpt_model(batch.unsqueeze(dim=0))
            pred_array = (
                torch.nn.functional.softmax(pred, dim=1)
                .argmax(dim=1)
                .detach()
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)
            writer.append_data(pred_array.squeeze())
    writer.close()


def _create_movie_probabilities(
    seq_path, root_folder, transform, gpt_model, sequence, save_name, device
):
    basename = os.path.basename(save_name)
    os.makedirs(basename, exist_ok=True)

    # Create a writer object
    writer = imageio.get_writer(
        save_name, fps=10
    )  # Adjust the fps as needed
    gpt_model.eval()
    with torch.no_grad():
        for idx, paths in enumerate(generate_paths(seq_path)):
            example = torch.stack(
                [
                    _convert_and_transform(
                        os.path.join(root_folder, sequence, ex), transform=transform
                    )
                    for ex in paths
                ]
            )
            # example needs to be shape b x t x (64*64), b=1,t=4
            batch = example.to(device)
            pred, _ = gpt_model(batch)
            pred_array = (
                torch.nn.functional.softmax(pred, dim=1)
                .argmax(dim=1)
                .detach()
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)
            writer.append_data(pred_array)
    writer.close()


class Flatten(object):

    def __call__(self, img_tensor):
        return torch.ravel(img_tensor)


def trained_default_config():
    trainer_cfg_node = utils.CfgNode()
    # device to train on
    trainer_cfg_node.device = "auto"
    # dataloder parameters
    trainer_cfg_node.num_workers = 4
    # optimizer parameters
    trainer_cfg_node.max_iters = None
    trainer_cfg_node.batch_size = 8
    trainer_cfg_node.learning_rate = 3e-4
    trainer_cfg_node.betas = (0.9, 0.95)
    trainer_cfg_node.weight_decay = 0.1  # only applied on matmul weights
    trainer_cfg_node.grad_norm_clip = 1.0
    return trainer_cfg_node


def dataset_default_config():
    C = utils.CfgNode()
    C.block_size = 4
    return C


def get_config():

    C = utils.CfgNode()

    # system
    C.system = utils.CfgNode()
    C.system.seed = 3407
    C.system.work_dir = "./out/worldmodel"
    C.system.experiment_id = ""

    # data
    C.data = dataset_default_config()
    C.data.dataset_dir = ""
    C.data.test_ratio = 0.1

    # model
    C.model = model.GPT.get_default_config()
    C.model.model_type = "gpt-nano"
    C.model.block_size = 4  # number of steps/images
    C.model.vocab_size = 16  # size image is compressed to 64*64->16

    # trainer
    C.trainer = trained_default_config()
    C.trainer.learning_rate = (
        5e-3  # the model we're using is so small that we can go a bit faster
    )
    C.trainer.batch_size = 64
    C.trainer.num_epochs = 2
    C.trainer.num_workers = 0

    return C


def _custom_sorter(fname):
    # use the number in the fname to sort
    return int(fname.split("_")[-1].split(".")[0])


def generate_paths(sequence_path):
    img_paths = sorted(os.listdir(sequence_path), key=_custom_sorter)
    # Loop through the sets of the sequence and auto-regressively generate the output
    # TODO: instead of just taking the argmax translate the probabilities of it being
    # a moving object to a range of 0-255 so we get probability of object maps
    start_index = 0
    end_index = 4
    while end_index < len(img_paths):
        yield img_paths[start_index:end_index]
        start_index += 1
        end_index += 1


def _convert_and_transform(fname, transform=None):
    img = Image.open(fname).convert("L")
    if transform:
        img = transform(img)
    return img


def main():
    cfg = get_config()
    cfg.merge_from_args(sys.argv[1:])
    if not cfg.data.dataset_dir:
        raise ValueError("You need to specify dataset directory using arguments --data.dataset_dir ")

    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if not cfg.system.experiment_id:
        experiment_id = f"experiment_ball_{current_time}"
    else:
        experiment_id = cfg.system.experiment_id
    print(f"Starting training {experiment_id}")

    
    os.makedirs(cfg.system.work_dir, exist_ok=True) # Check we are able to save results to appropriate location

    # Define transformations to apply to the images
    transform = transforms.Compose(
        [transforms.ToTensor(), Flatten()]  # Convert the image to a PyTorch tensor
    )

    # Create train/val split
    dset = dataset.CustomDataset(cfg.data.dataset_dir, transform=transform)

    data_idxs = np.array(list(range(len(dset))))
    np.random.seed(cfg.system.seed)
    val_idxs = np.random.choice(
        data_idxs, size=np.floor(cfg.data.test_ratio * len(dset)).astype(np.int64), replace=False
    )
    train_idxs = np.setdiff1d(data_idxs, val_idxs, assume_unique=True)
    train_dset = torch.utils.data.Subset(dset, train_idxs)
    validation_dset = torch.utils.data.Subset(dset, val_idxs)

    # When training the images need to be converted to tensors
    # The shape of the input is Batch x Timestemps x embedding size
    # Each image gets translated by MLP which takes input 64x64 and outputs size embedding (16 etc.)
    # Transformer operates on this embedding plus positional embedding
    # The output of the transformet is Batch x Timestamps x embedding size
    # This gets translated from embedding size back to 64x64

    gpt_model = model.GPT(cfg.model)

    model_trainer = train.Trainer(
        cfg.trainer, gpt_model, train_dset, validation_dataset=validation_dset
    )

    model_trainer.run()

    print(f"Saving model to {cfg.system.work_dir}")
    savelocation = os.path.join(cfg.system.work_dir, experiment_id, "checkpoints")
    os.makedirs(savelocation, exist_ok=True) # Check we are able to save results to appropriate location
    torch.save(gpt_model.state_dict(), os.path.join(savelocation, "trained_model.pt"))
    print(f"Model saved at {os.path.join(savelocation, 'trained_model.pt')}")



def generate_movie(checkpoint_path:str, dataset_dir:str, save_dir:str):

    # Fix issue with rich https://stackoverflow.com/questions/71143520/python-rich-restore-cursor-default-values-on-exit
    atexit.register(lambda: print("\x1b[?25h"))  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config()
    gpt_model = model.GPT(cfg.model)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    # Define transformations to apply to the images
    transform = transforms.Compose(
        [transforms.ToTensor(), Flatten()]  # Convert the image to a PyTorch tensor
    )
    gpt_model.load_state_dict(torch.load(checkpoint_path))
    gpt_model.to(device)
    
    progress = Progress(
        SpinnerColumn("bouncingBall"),
        TextColumn("{task.description}  {task.completed} of {task.total}"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    
    
    sequences = sorted(os.listdir(dataset_dir))
    os.makedirs(save_dir, exist_ok=True)
    generate_task = progress.add_task("[green]Generating movies...", total=len(sequences))
    progress.start()
    try:

        for idx,sequence in enumerate(sequences):
            save_name = os.path.join(save_dir, f"sequence_{idx}.mp4")
            seq_path = os.path.join(dataset_dir,sequence)
            progress.update(generate_task, description=f"[green]Generating {save_name}...")

            pred_gen = _prediction_generator(seq_path, dataset_dir, transform, gpt_model, sequence, device)
            gt_gen = _ground_truth_generator(seq_path, dataset_dir, sequence)
            
            movies.generate_movie(gt_gen, pred_gen, save_name)
            # _create_movie(
            #     seq_path, dataset_dir, transform, gpt_model, sequence, save_name, device
            # )
            progress.update(generate_task, advance=1)
        progress.stop()

    except KeyboardInterrupt: # Restore appropriate state for rich
        progress.stop()

def cli():
    fire.Fire({
        "movie": generate_movie,
    })
