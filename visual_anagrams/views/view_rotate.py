from PIL import Image

import torch

from .view_base import BaseView


class Rotate90CWView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        # Quarter turns are exact permutations. An affine sampling grid can
        # lose pixel coordinates under CUDA BF16 autocast at 1024px, breaking
        # inverse alignment and leaving noise around the central region.
        return torch.rot90(im, k=-1, dims=(-2, -1))

    def inverse_view(self, noise):
        return torch.rot90(noise, k=1, dims=(-2, -1))

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * -90

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame


class Rotate90CCWView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return torch.rot90(im, k=1, dims=(-2, -1))

    def inverse_view(self, noise):
        return torch.rot90(noise, k=-1, dims=(-2, -1))

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * 90

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame


class Rotate180View(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return torch.rot90(im, k=2, dims=(-2, -1))

    def inverse_view(self, noise):
        return torch.rot90(noise, k=2, dims=(-2, -1))

    def make_frame(self, im, t):
        im_size = im.size[0]
        frame_size = int(im_size * 1.5)
        theta = t * 180

        frame = Image.new('RGB', (frame_size, frame_size), (255, 255, 255))
        centered_loc = (frame_size - im_size) // 2
        frame.paste(im, (centered_loc, centered_loc))
        frame = frame.rotate(theta, 
                             resample=Image.Resampling.BILINEAR, 
                             expand=False, 
                             fillcolor=(255,255,255))

        return frame
