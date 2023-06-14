import torch
import random
from typing import Any
from types import FunctionType
from corruption.corruptions import *

corruption_function_RGB = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                           glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
                           elastic_transform, pixelate, jpeg_compression, speckle_noise,
                           gaussian_blur, spatter, saturate, rain]  # 20

# All the same for IR except that brightness is used to simulate saturation
# (so saturate function does not appear the list but brightess appears instead)
corruption_function_IR = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                          glass_blur, motion_blur, zoom_blur, snow, frost, fog, contrast,
                          elastic_transform, pixelate, jpeg_compression, speckle_noise,
                          gaussian_blur, spatter, brightness, rain, none]  # 20

not_correlated_corruption_function_RGB = [brightness, saturate, contrast, gaussian_noise, shot_noise,
                                          impulse_noise, speckle_noise, pixelate, jpeg_compression, elastic_transform]
not_correlated_corruption_function_IR = [brightness, contrast, gaussian_noise, shot_noise, impulse_noise,
                                         speckle_noise, pixelate, jpeg_compression, elastic_transform, none]

# Specific compose class to handle multimodal data transformations
class Compose:
    def __init__(self, transforms, CIL, XPATCH, scenario_eval="normal"):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = transforms
        self.XPATCH = XPATCH
        self.CIL = CIL
        self.scenario_eval = scenario_eval
    def __call__(self, img_RGB, img_IR=None):
        for t in self.transforms:
            if "mixing_erasing" in str(t):
                if t.type == "soft":
                    img_RGB, img_IR = t(img_RGB), img_IR
                if t.type == "soft_IR":
                    img_RGB, img_IR = img_RGB, t(img_IR)
                if t.type == "self" and (self.XPATCH == "S-PATCH" or self.CIL):
                    img_RGB, img_IR = t(img_RGB), img_IR
                if t.type == "self" and self.XPATCH == "MS-PATCH":
                    img_RGB, img_IR = t(img_RGB), t(img_IR)
            elif self.scenario_eval != "normal" and "Corruption_transform" in str(t):
                if self.scenario_eval == "C":  # Apply corrupt on RGB only
                    img_RGB, img_IR = t(img_RGB, modality="RGB"), img_IR
                elif self.scenario_eval == "UCD": # Apply corrupt on both RGB and IR independantly (Uncorrelated)
                    img_RGB, img_IR = t(img_RGB, modality="RGB"), t(img_IR, modality="IR")
                elif self.scenario_eval == "CCD": # Apply corrupt on both RGB and IR with eventual correlations
                    img_RGB, img_IR = t(img_RGB, img_IR)
            else :
                try : # Handle most of the transformations like toTensor, horizontal flips, crops etc..
                    img_RGB = t(img_RGB)
                    img_IR = t(img_IR)
                except:  # Handle and multimodal patch mixing
                    img_RGB, img_IR = t(img_RGB, img_IR)

        return img_RGB, img_IR

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

# Adapted from https://github.com/MinghuiChen43/CIL-ReID/blob/main/datasets/make_dataloader.py
# => UCD
class uncorrelated_corruption_transform(object):
    def __init__(self, level=0, type='all'):
        self.level = level
        self.type = type

        self.corruption_function_RGB = corruption_function_RGB  # if modality == "RGB" else corruption_function_IR
        self.corruption_function_IR = corruption_function_IR  # if modality == "RGB" else corruption_function_IR

        self.rng = random  # local random seed

    def __call__(self, img, modality="RGB"):
        corruption_function = self.corruption_function_RGB if modality == "RGB" else self.corruption_function_IR
        if self.level > 0 and self.level < 6:
            level_idx = self.level
        else:
            level_idx = self.rng.choice(range(1, 6))
        if self.type == 'all':
            corrupt_func = self.rng.choice(corruption_function)
        else:
            func_name_list = [f.__name__ for f in corruption_function]
            corrupt_idx = func_name_list.index(self.type)
            corrupt_func = corruption_function[corrupt_idx]

        c_img = corrupt_func(img.copy(), severity=level_idx, modality=modality)
        img = Image.fromarray(np.uint8(c_img))
        return img

### CCD
class correlated_corruption_transform(object):
    def __init__(self, level=0, type='all', CCD_X=0):
        self.level = level
        self.type = type

        self.corruption_function = [corruption_function_RGB, corruption_function_IR]

        self.not_correlated_function = [not_correlated_corruption_function_RGB, not_correlated_corruption_function_IR]
        self.not_correlated_function_names = [[f.__name__ for f in not_correlated_corruption_function_RGB],
                                              [f.__name__ for f in not_correlated_corruption_function_IR]]

        self.rng = random
        self.weather_related_corruptions = ["fog", "frost", "snow", "rain"]

        self.CCD_X = CCD_X  # Uncorrelated hetero modality have X% chance of being clean

    def __call__(self, img_RGB, img_IR):

        img_list = [img_RGB, img_IR]
        modality = ["RGB", "IR"]
        # Determine from which modality we select the corruption first
        modality_selection = self.rng.choice([[0, 1], [1, 0]])
        level_idx, c_img, corrupt_func = [0, 0], [0, 0], [0, 0]

        if self.level > 0 and self.level < 6:
            level_idx[modality_selection[0]] = self.level
        else: # default
            level_idx[modality_selection[0]] = self.rng.choice(range(1, 6))

        if self.type == 'all': # Default

            # Random modality corruption for the first modality
            corrupt_func[modality_selection[0]] = self.rng.choice(self.corruption_function[modality_selection[0]])
            corrupt_name = corrupt_func[modality_selection[0]].__name__

            # Then, corrupting regarding the corrupt name. If not correlated:
            if corrupt_name in self.not_correlated_function_names[modality_selection[0]]:
                if self.rng.random() < self.CCD_X / 100 : # X percent chance of being clean
                    corrupt_func[modality_selection[1]] = none
                else:
                    corrupt_func[modality_selection[1]] = self.rng.choice(
                        self.not_correlated_function[modality_selection[1]])
                level_idx[modality_selection[1]] = self.rng.choice(range(1, 6))

            # If the hetero modality must be correlated
            else:
                corrupt_func[modality_selection[1]] = corrupt_func[modality_selection[0]]

                # For weather related corruption => Same degradation apply
                if corrupt_name in self.weather_related_corruptions:
                    level_idx[modality_selection[1]] = level_idx[modality_selection[0]]
                # For motion blur, level is selected higher or equal for IR
                elif corrupt_name == "motion_blur":
                    i = 1 if modality_selection[
                                 0] == 0 else -1  # depending on the selected modality we upscale or reduce level
                    if i == 1:
                        k = self.rng.randint(0, 5 - level_idx[modality_selection[0]])
                    else:
                        k = self.rng.randint(0, level_idx[modality_selection[0]] - 1)

                    level_idx[modality_selection[1]] = level_idx[modality_selection[0]] + i * k
                # For any other blur, level can be different
                elif corrupt_name in ["spatter", "glass_blur", "defocus_blur", "zoom_blur", "gaussian_blur"]:
                    level_idx[modality_selection[1]] = self.rng.choice(
                        range(1, 6))  # Can be different level from RGB to IR for spatter

        else: # In the case it is a precise corruption request
            func_name_list = [f.__name__ for f in self.corruption_function]
            corrupt_idx = func_name_list.index(self.type)
            corrupt_func = self.corruption_function[corrupt_idx]

        # Applying the corruption transforms
        c_img[modality_selection[0]] = corrupt_func[modality_selection[0]](img_list[modality_selection[0]].copy(),
                                                                           severity=level_idx[modality_selection[0]],
                                                                           modality=modality[modality_selection[0]])

        c_img[modality_selection[1]] = corrupt_func[modality_selection[1]](img_list[modality_selection[1]].copy(),
                                                                           severity=level_idx[modality_selection[1]],
                                                                           modality=modality[modality_selection[1]])
        img_RGB = Image.fromarray(np.uint8(c_img[modality_selection[0]])) if modality_selection[
                                                                                 0] == 0 else Image.fromarray(
            np.uint8(c_img[modality_selection[1]]))
        img_IR = Image.fromarray(np.uint8(c_img[modality_selection[0]])) if modality_selection[
                                                                                0] == 1 else Image.fromarray(
            np.uint8(c_img[modality_selection[1]]))
        return img_RGB, img_IR

# Masking DA
class Masking(object):
    def __init__(self, ratio=0.5):

        self.ratio = ratio  # Probability that one pair of images contain a masked img
        self.rng = random.Random()  # local random seed

    def __call__(self, img_RGB, img_IR):

        img_list = [img_RGB, img_IR]
        modality_selection = self.rng.choice([[0, 1], [1, 0]])  # Determine from which modality we apply the corruption first
        # modality_selection = [0,1]  # Determine from which modality we apply the corruption first

        if self.rng.random() < 0.5:
            if self.rng.random() < self.ratio/2 :
                img_list[modality_selection[0]] = Image.new(mode="RGB", size=(144, 288), color=(255, 255, 255))
                return img_list # Two masked images never happened, we avoid having two blanked images.

        if self.rng.random() < 0.5:
            if self.rng.random() < self.ratio/2 :
                img_list[modality_selection[1]] = Image.new(mode="RGB", size=(144, 288), color=(255, 255, 255))

        return img_list

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")