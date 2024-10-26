import torch, os, math, warnings
from PIL import Image
from typing import *
import imageio
import numpy as np

class Visualizer(object):
    def __init__(self, config):
        self.gif_dir = config.gif_dir

        pass
    def obs_to_image(self, obs: torch.Tensor):
        if obs.shape[-3] == 1:
            return obs.expand(*obs.shape[:-3], 3, *obs.shape[-2:])
        else:
            assert obs.shape[-3] % 3 == 0
            return obs.reshape(*obs.shape[:-3], 3, -1, obs.shape[-1])

    def save_image(self, uint8_image: torch.Tensor, fp: str):
        im = Image.fromarray(uint8_image.permute(1, 2, 0).numpy())
        im.save(fp)

    @torch.no_grad()
    def collect_frames(self, obs, rssm_state, rssm_state_0, RSSM, ObsDecoder, video_frames_dict):
        # Collect real vs. predicted frames for video
        rssm_state_dict = {"rssm_state":rssm_state}
        for key, value in rssm_state_dict.items():
            model_state = RSSM.get_model_state(value)
            reconstruction: torch.Tensor = ObsDecoder(model_state).mean
            video_frames_dict[key].append(
                self.make_grid(
                    torch.cat([
                        self.obs_to_image(torch.unsqueeze(obs, dim=0).cpu()),
                        self.obs_to_image(reconstruction.cpu())
                        ], dim=-1).add_(0.5),
                    nrow=1
                ).mul_(255).clamp_(0, 255).to(torch.uint8)
            )
            
    @torch.no_grad()
    def collect_frames_cartpole(self, obs, rssm_state, rssm_state_0, RSSM, ObsDecoder, video_frames_dict):
        # Collect real vs. predicted frames for video        
        rssm_state_dict = {"rssm_state":rssm_state}
        
        video_frames_dict["obs"].append(self.obs_to_image(obs.cpu().add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8)))
        for key, value in rssm_state_dict.items():
            model_state = RSSM.get_model_state(value)
            reconstruction: torch.Tensor = ObsDecoder(model_state).mean
            video_frames_dict[key].append(self.obs_to_image(reconstruction.squeeze().cpu()).add_(0.5).mul_(255).clamp_(0, 255).to(torch.uint8))

    def write_video(self, frames: List[torch.Tensor], title, path="", fps=30):
        with imageio.get_writer(os.path.join(path, "%s.mp4" % title), mode='I', fps=fps) as writer:
            for frame in frames:
                # VideoWrite expects H x W x C in BGR
                writer.append_data(frame.permute(1, 2, 0).numpy())


    # make_grid but ensures divisible by 16 (for imageio ffmpeg)
    # adapted from https://github.com/pytorch/vision/blob/eac3dc7bab436725b0ba65e556d3a6ffd43c24e1/torchvision/utils.py#L14
    @torch.no_grad()
    def make_grid(
        self,
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        ensure_divisble_by: int = 16,
        **kwargs,
    ) -> torch.Tensor:
        """
        Make a grid of images.
        Args:
            tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
                or a list of images all of the same size.
            nrow (int, optional): Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding (int, optional): amount of padding. Default: ``2``.
            normalize (bool, optional): If True, shift the image to the range (0, 1),
                by the min and max values specified by ``value_range``. Default: ``False``.
            value_range (tuple, optional): tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each (bool, optional): If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        Returns:
            grid (Tensor): the tensor containing grid of images.
        """
        if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

        if "range" in kwargs.keys():
            warning = "range will be deprecated, please use value_range instead."
            warnings.warn(warning)
            value_range = kwargs["range"]

        # if list of tensors, convert to a 4D mini-batch Tensor
        if isinstance(tensor, list):
            tensor = torch.stack(tensor, dim=0)

        if tensor.dim() == 2:  # single image H x W
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 3:  # single image
            if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
                tensor = torch.cat((tensor, tensor, tensor), 0)
            tensor = tensor.unsqueeze(0)

        if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
            tensor = torch.cat((tensor, tensor, tensor), 1)

        if normalize is True:
            tensor = tensor.clone()  # avoid modifying tensor in-place
            if value_range is not None:
                assert isinstance(
                    value_range, tuple
                ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

            def norm_ip(img, low, high):
                img.clamp_(min=low, max=high)
                img.sub_(low).div_(max(high - low, 1e-5))

            def norm_range(t, value_range):
                if value_range is not None:
                    norm_ip(t, value_range[0], value_range[1])
                else:
                    norm_ip(t, float(t.min()), float(t.max()))

            if scale_each is True:
                for t in tensor:  # loop over mini-batch dimension
                    norm_range(t, value_range)
            else:
                norm_range(tensor, value_range)

        assert isinstance(tensor, torch.Tensor)
        if tensor.size(0) == 1:
            return tensor.squeeze(0)

        # make the mini-batch of images into a grid
        nmaps = tensor.size(0)
        xmaps = min(nrow, nmaps)
        ymaps = int(math.ceil(float(nmaps) / xmaps))
        height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
        num_channels = tensor.size(1)
        full_grid_height = height * ymaps + padding
        if full_grid_height % ensure_divisble_by != 0:
            full_grid_height += ensure_divisble_by - (full_grid_height % ensure_divisble_by)
        full_grid_width = width * xmaps + padding
        if full_grid_width % ensure_divisble_by != 0:
            full_grid_width += ensure_divisble_by - (full_grid_width % ensure_divisble_by)
        grid = tensor.new_full((num_channels, full_grid_height, full_grid_width), pad_value)
        k = 0
        for y in range(ymaps):
            for x in range(xmaps):
                if k >= nmaps:
                    break
                # Tensor.copy_() is a valid method but seems to be missing from the stubs
                # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
                grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                    2, x * width + padding, width - padding
                ).copy_(tensor[k])
                k = k + 1
        return grid

    @torch.no_grad()
    def output_video(self, step, e, video_frames_dict):
        vis_dir = os.path.join(self.gif_dir, f'{step}')
        os.makedirs(vis_dir, exist_ok=True)
        print(vis_dir)
        for key, value in video_frames_dict.items():
            self.write_video(value, f"episode_{e}_{key}", vis_dir)  # Lossy compression
            self.save_image(
                torch.as_tensor(value[-1]),
                os.path.join(vis_dir, f"episode_{e}_{key}.png"),
            )
        print('Saved visualization.')

    @torch.no_grad()
    def output_picture(self, save_dir, e, video_frames_dict):
        os.makedirs(save_dir, exist_ok=True)
        final_dist = {}
        for key, value in video_frames_dict.items():
            concatenated_frames = np.hstack([frame.permute(1, 2, 0).numpy() for frame in value])
            final_dist[key] = concatenated_frames
            filename = os.path.join(save_dir, f"episode_{e}_{key}.png")
            result_img = Image.fromarray(concatenated_frames)
            result_img.save(filename)
        # result = [final_dist['obs'], final_dist['rssm_state_1'], final_dist['rssm_state_2'], final_dist['rssm_state_3'], final_dist['rssm_state_4']]
        result = [final_dist['obs'], final_dist['rssm_state_12'], final_dist['rssm_state_3'], final_dist['rssm_state_34']]
        final_frames = np.vstack(result)
        result_img = Image.fromarray(final_frames)
        filename = os.path.join(save_dir, f"result_{e}.png")
        result_img.save(filename)
        print('Saved visualization.')
        
    @torch.no_grad()
    def output_picture_dmc(self, save_dir, e, video_frames_dict, kind=['obs', '1', '12', '3', '4', '34']):
        os.makedirs(save_dir, exist_ok=True)
        final_dist = {}
        for key, value in video_frames_dict.items():
            concatenated_frames = np.hstack([frame.permute(1, 2, 0).numpy() for frame in value])
            final_dist[key] = concatenated_frames
            filename = os.path.join(save_dir, f"episode_{e}_{key}.png")
            result_img = Image.fromarray(concatenated_frames)
            result_img.save(filename)
        # result = [final_dist['obs'], final_dist['rssm_state_1'], final_dist['rssm_state_2'], final_dist['rssm_state_3'], final_dist['rssm_state_4']]
        result = [final_dist[key] for key in kind]
        final_frames = np.vstack(result)
        result_img = Image.fromarray(final_frames)
        filename = os.path.join(save_dir, f"result_{e}.png")
        result_img.save(filename)
        print('Saved visualization.')