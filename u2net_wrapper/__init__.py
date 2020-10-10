import tempfile
from enum import Enum
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from skimage import io, transform
from torchvision import transforms
from torch.autograd import Variable

from u2net_wrapper.utils import (
    download_file,
    get_md5_from_file
)

from u2net_wrapper.contrib.model import U2NET, U2NETP

class RescaleImage:
    """
    Transformation that rescales image data.

    Adapted from the U-2-net data loader source code.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        """Perform the transformation."""
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        output = transform.resize(
            image,
            (self.output_size, self.output_size),
            mode='constant'
        )

        return output

class ImageToTensor:
    """
    Convert image to float tensors.

    Adapted from the U-2-net data loader source.
    """
    def __init__(self, colour_space=0):
        """
        Initialize the transformation.

        :param colour_space:
            An integer denoting the colour space to transform the image into.
            A value of 0 means RGB, 1 means Lab, and 2 means Lab + RGB.
        """

        self.colour_space = colour_space

    def __call__(self, image):
        """Perform the transformation."""
        # Change the color space
        if self.colour_space == 2: # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0],image.shape[1],6))
            tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
            if image.shape[2]==1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

        elif self.colour_space == 1: # with Lab color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))

            if image.shape[2]==1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

        else: # with rgb color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            image = image/np.max(image)
            if image.shape[2]==1:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
            else:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpImg = tmpImg.transpose((2, 0, 1))
        return torch.from_numpy(tmpImg)

class U2Net:
    """An interface for the U-2-net model."""
    def __init__(self, pretrained_model_name='large',
                 checkpoint_path=None):
        """
        Initialize the U-2-net model.

        :param pretrained_model_name:
            The name of the pretrained model. Defaults to 'large'.
        :param checkpoint_path:
            The model checkpoint to restore from. Defaults to None.
        """

        if checkpoint_path is None and pretrained_model_name is None:
            raise ValueError('Both checkpoint_path and pretrained_checkpoint_type cannot be none!')

        if checkpoint_path is None:
            checkpoint_path = U2Net._get_pretrained_checkpont(pretrained_model_name)

        if pretrained_model_name == 'small':
            self._model = U2NETP(in_ch=3, out_ch=1)
        else:
            self._model = U2NET(in_ch=3, out_ch=1)

        self._model.load_state_dict(torch.load(checkpoint_path))
        if torch.cuda.is_available():
            self._model.cuda()

        self._model.eval()

    @staticmethod
    def _normalize_prediction(prediction):
        """Normalize the output of the model."""
        hi = torch.max(prediction)
        lo = torch.min(prediction)

        normalized = (prediction - lo) / (hi - lo)
        return normalized

    def segment_image(self, source_filepath, destination_filepath=None):
        """
        Segment an image

        :param source_filepath:
            The path to the image file to segment.
        :param destination_filepath:
            The path to save the output. Defaults to None.
        :returns:
            A PIL Image object representing the segmentation map.
        """
        source_filepath = str(source_filepath)
        input_image, input_image_dim = U2Net._load_image(source_filepath)
        input_image = input_image.unsqueeze(0).type(torch.FloatTensor)

        if torch.cuda.is_available():
            input_image = Variable(input_image.cuda())
        else:
            input_image = Variable(input_image)

        outputs = self._model(input_image)
        segmentation_map = U2Net._normalize_prediction(outputs[0][:, 0, :, :])

        # Convert data to cpu
        segmentation_map = segmentation_map.squeeze().cpu().data.numpy()

        # Load and resize output image (to size of source)
        segmentation_map = Image.fromarray(segmentation_map * 255).convert('RGB')
        segmentation_map = segmentation_map.resize(
            input_image_dim,
            resample=Image.BILINEAR
        )

        if destination_filepath is not None:
            segmentation_map.save(str(destination_filepath))

        # Delete outputs (to avoid memory leak?)
        del outputs

        return segmentation_map

    def remove_background(self, source_filepath, destination_filepath=None,
                          threshold=0.9, rescale_amount=255):
        """
        Removes the background of an image.

        :param source_filepath:
            The path to the image file to segment.
        :param destination_filepath:
            The path to save the output. Defaults to None.
        :param threshold:
            Threshold to keep pixel. Defaults to 0.9.
        :param rescale_amount:
            The max value of an element in the image data (used to normalize
            the data into a 0 to 1 range). Defaults to 255.0 (for RGB colour space).
        :returns:
            A PIL Image object representing the image with its background removed.
        """
        output = np.asarray(self.segment_image(source_filepath), dtype=np.float)
        output = output / rescale_amount

        # Convert segmentation map into a binary mask
        output[output > threshold] = 1
        output[output <= threshold] = 0

        output_shape = output.shape
        a_layer_init = np.ones(shape=(output_shape[0], output_shape[1], 1))
        mul_layer = np.expand_dims(output[:,:,0], axis=2)
        a_layer = mul_layer * a_layer_init
        rgba_out = np.append(output, a_layer, axis=2)

        # Map to original image
        input_image = np.asarray(Image.open(str(source_filepath)),  dtype=np.float)
        input_image = input_image / rescale_amount

        if input_image.shape[2] == 4:
            rgba_inp = np.array(input_image)
            rgba_inp[..., 3] = 1
        else:
            a_layer = np.ones(shape=(output_shape[0], output_shape[1], 1))
            rgba_inp = np.append(input_image, a_layer, axis=2)

        processed_image = rgba_inp * rgba_out * rescale_amount
        processed_image = Image.fromarray(processed_image.astype('uint8'), 'RGBA')
        if destination_filepath is not None:
            processed_image.save(str(destination_filepath))

        return processed_image

    @staticmethod
    def _get_pretrained_checkpont(pretrained_model_name):
        """Gets a pretrained model checkpoint."""
        # We shouldn't be hardcoding this information, but it will work for now.
        _PRETRAINED_CHECKPOINT_URLS = {
            'large': {
                'md5': '347c3d51b01528e5c6c071e3cff1cb55',
                'url': 'http://136.243.107.6/u2net-wrapper/1.0.0/u2net.pth',
                'save_filename': 'u2net_large.pth'
            },
            'small': {
                'md5': 'e4f636406ca4e2af789941e7f139ee2e',
                'url': 'http://136.243.107.6/u2net-wrapper/1.0.0/u2netp.pth',
                'save_filename': 'u2net_small.pth'
            }
        }

        destination_dir = Path(tempfile.gettempdir()) / 'u2net-wrapper'
        destination_dir.mkdir(exist_ok=True)

        pretrained_info = _PRETRAINED_CHECKPOINT_URLS[pretrained_model_name]
        destination_filepath = destination_dir / pretrained_info['save_filename']
        if get_md5_from_file(destination_filepath) != pretrained_info['md5']:
            print(f'Downloading U-2-net-{pretrained_model_name} checkpoint...')
            download_file(pretrained_info['url'], destination_filepath)

        return destination_filepath

    @staticmethod
    def _load_image(filepath):
        """Load an image from the given filepath into a format ready for the U-2-net model."""
        image = io.imread(str(filepath))
        dim = (image.shape[1], image.shape[0])

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        image_transform = transforms.Compose([
            RescaleImage(320),
            ImageToTensor(colour_space=0)
        ])

        return image_transform(image), dim