
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

def _custom_sorter(fname):
  # use the number in the fname to sort
  return int(fname.split("_")[-1].split(".")[0])


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.sequences = sorted(os.listdir(data_dir))  #
        
        self._data_paths = list(self._create_sequences())

    def _create_sequences(self):
        # Create a sequence of 4 input images and predict the 5th
        for seq in self.sequences:
          
          seq_path = os.path.join(self.data_dir,seq)
          img_paths = sorted(os.listdir(seq_path), key = _custom_sorter) # sort by number
          seq_img_paths = [os.path.join(seq_path, img) for img in img_paths]
          for pred_idx in range(4,len(seq_img_paths)):
            yield (seq_img_paths[pred_idx-4:pred_idx], seq_img_paths[pred_idx])

    def __len__(self):
        return len(self._data_paths)

    def _convert_and_transform(self, fname):
      img = Image.open(fname).convert('L')
      if self.transform:
        img = self.transform(img)
      return img

    def __getitem__(self, idx):
        data = self._data_paths[idx]
        x = [self._convert_and_transform(ex) for ex in data[0]]
        y = self._convert_and_transform(data[1])
        return (x,y)
        # img_path = self.img_paths[idx]
        # img = Image.open(img_path).convert('RGB')
        # label = self.class_to_idx[os.path.basename(os.path.dirname(img_path))]  # Extract label from path
        # if self.transform:
        #     img = self.transform(img)
        # return img, label


def collate_fn(batch):
  inpt = []
  target = []
  for ex in batch:
    inpt.append(torch.stack(ex[0]))
    target.append(ex[1])
  
  return torch.stack(inpt), torch.stack(target)