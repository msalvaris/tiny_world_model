import os

import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import fire
from datetime import datetime
import sys

from world_model import dataset, model, train, utils


def _create_movie(
    seq_path, root_folder, transform, gpt_model, sequence, save_name, device
):
    
    basename = os.path.basename(save_name)
    os.makedirs(basename, exist_ok=True)

    # Create a writer object
    writer = imageio.get_writer(
        save_name, fps=10
    )  # Adjust the fps as needed
    # output_path = os.path.join("/content","pred_sequences",sequence)
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
            # img = Image.fromarray(pred_array)
            # img.save(os.path.join(output_path, f"frame_{idx+4}.png")) # Offset by 4 as that is the first sequence
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
            # torch.nn.functional.softmax(pred,dim=1)[]
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

    # data
    C.data = dataset_default_config()

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
        # print(img_paths[start_index:end_index])
        yield img_paths[start_index:end_index]
        start_index += 1
        end_index += 1


def _convert_and_transform(fname, transform=None):
    img = Image.open(fname).convert("L")
    if transform:
        img = transform(img)
    return img


def main(root_folder:str, save_folder:str):
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_id = f"experiment_ball_{current_time}"
    print(f"Starting training {experiment_id}")

    
    
    os.makedirs(save_folder, exist_ok=True) # Check we are able to save results to appropriate location

    # Define transformations to apply to the images
    transform = transforms.Compose(
        [transforms.ToTensor(), Flatten()]  # Convert the image to a PyTorch tensor
    )

    # Create train/val split
    dset = dataset.CustomDataset(root_folder, transform=transform)

    data_idxs = np.array(list(range(len(dset))))
    test_ratio = 0.1
    np.random.seed(42)
    val_idxs = np.random.choice(
        data_idxs, size=np.floor(test_ratio * len(dset)).astype(np.int64), replace=False
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

    cfg = get_config()
    print(sys.argv[3:])
    cfg.merge_from_args(sys.argv[3:])
    gpt_model = model.GPT(cfg.model)

    model_trainer = train.Trainer(
        cfg.trainer, gpt_model, train_dset, validation_dataset=validation_dset
    )

    model_trainer.run()

    print(f"Saving model to {save_folder}")
    savelocation = os.path.join(save_folder, experiment_id, "checkpoints")
    os.makedirs(savelocation, exist_ok=True)# Check we are able to save results to appropriate location
    torch.save(gpt_model.state_dict(), os.path.join(savelocation, "initial_model.pt"))

   

def generate_movie(checkpoint_path:str, root_folder:str, save_name:str):
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
    
    sequences = sorted(os.listdir(root_folder))
    seq_path = os.path.join(root_folder,sequences[0])
    gpt_model.load_state_dict(torch.load(checkpoint_path))
    
    gpt_model.to(device)

    _create_movie(
        seq_path, root_folder, transform, gpt_model, sequences[0], save_name, device
    )


def cli():
    fire.Fire({
        "ball": main,
        "movie": generate_movie,
    })
