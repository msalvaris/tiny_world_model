import numpy as np
from PIL import Image
import torch


def combine_images(image_list, rows=None, cols=None, padding=10, background_color=(255, 255, 255)):
  if not rows and not cols:
      raise ValueError("You must specify either the number of rows or columns.")

  if rows and cols:
      raise ValueError("You can only specify either the number of rows or columns, not both.")

  if rows:
      cols = -(-len(image_list) // rows)  # ceiling division to calculate number of columns
  else:
      rows = -(-len(image_list) // cols)  # ceiling division to calculate number of rows

  max_width = max(image.width for image in image_list)
  max_height = max(image.height for image in image_list)

  total_width = cols * max_width + (cols - 1) * padding
  total_height = rows * max_height + (rows - 1) * padding

  new_image = Image.new("RGB", (total_width, total_height), background_color)

  current_x = 0
  current_y = 0

  for image in image_list:
      new_image.paste(image, (current_x, current_y))
      current_x += max_width + padding
      if current_x >= total_width:
          current_x = 0
          current_y += max_height + padding

  return new_image


# def convert_to_image(pred,target):
#   b,_ = pred.shape
#   pred_array = (pred.view(b,64,64).detach().cpu().numpy()*255).astype(np.uint8)
#   target_array = (target.view(b,64,64).detach().cpu().numpy()*255).astype(np.uint8)
#   for i in range(b):
#     yield combine_images([Image.fromarray(pred_array[i,:]),Image.fromarray(target_array[i,:])], cols=2)


def convert_to_image(pred,target):
  b,_,_,_= pred.shape
  pred_array = (torch.nn.functional.softmax(pred,dim=1).argmax(dim=1).detach().cpu().numpy()*255).astype(np.uint8)
  target_array = (target.view(b,64,64).detach().cpu().numpy()*255).astype(np.uint8)
  for i in range(b):
    yield combine_images([Image.fromarray(pred_array[i,:]),Image.fromarray(target_array[i,:])], cols=2)



  
class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
