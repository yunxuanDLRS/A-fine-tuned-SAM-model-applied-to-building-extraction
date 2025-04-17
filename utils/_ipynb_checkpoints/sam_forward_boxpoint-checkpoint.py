from segment_anything.modeling.sam import Sam
from utils.functions import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import random


def SamForward(sam: Sam, img: torch.FloatTensor, mask_label: torch.FloatTensor=None, return_logits: bool = False, numpy: bool = False, multimask_output: bool = False, device='cuda', return_prompt: bool = False, num_points: int = 1, prompt_box:bool=None) -> torch.FloatTensor:
    """
    Prompt inputs are generated from a single pixel from mask label.

    Args:
        img (torch.FloatTensor): RGB image of torch.float32 type tensor with shape (N,H,W,C)
        mask_label (torch.FloatTensor): Mask label of torch.float32 type tensor with shape (N,H,W). Prompt inputs are generated from this mask label.
        return_logits (bool, optional): If True, output masks are thresholded to binary values. Turn off when .backward() call.
        numpy(bool, optional): If true, predicted masks are converted to CPU NumPy arrays.
        multimask_output(bool, optional): If true, output masks are three masks with different resolutions. If false, output masks are single mask with the same resolution as input image (the first, coarse mask returned only).
        return_prompt(bool, optional): Returns randomly sampled prompt input if true
        num_points(int, optional): Number of randomly sampled prompt points from mask label. Defaults to 1.
        prompt_points:如果直接输入prompt_points就可以不用再生成点，直接用所输入的
    Returns:
        masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
        'iou_predictions': (torch.Tensor) The model's predictions of mask quality, in shape BxC.
        'low_res_logits': (torch.Tensor) Low resolution logits with shape BxCxHxW, where H=W=256.
                        Can be passed as mask input to subsequent iterations of prediction.
    """
    # SAM FORWARD
    with torch.no_grad():

        # 1. Image Encoder Forward
        # *********************************************************************************************************
        batch_size = img.shape[0]
        original_size = img.shape[1:3]                        # (H_org,W_org)
        input_image = apply_image(img, sam=sam).to(device)    # Expects a torch tensor NxHxWxC in uint8 format.预处理.
        input_size = tuple(input_image.shape[-2:])            # (H_in,W_in)
        input_image = sam.preprocess(input_image)             # Normalize pixel values and pad to a square input.
        image_embeddings = sam.image_encoder(input_image)
        # *********************************************************************************************************

        # 2. Create a random point prompt from mask_label and Prompt Encoder Forward
        # *********************************************************************************************************
        if mask_label is not None:
            prompt_box = []
            prompt_points = []
            for i in range(batch_size):                                        # 循环处理每一张图像
                prompt_point_indices = torch.argwhere(mask_label[i] == 1)    # 选出图像中的前景目标点位
                # ..................................point............................
                num_points = random.randint(1, 4)
                sampled_indices = torch.randint(prompt_point_indices.shape[0], size=(num_points,))
                prompt_point = torch.flip(prompt_point_indices[sampled_indices], (1,))
                prompt_points.append(prompt_point)
                # ...................................box............................
                # 获取最大纵坐标值的索引
                max_y_index = torch.argmax(prompt_point_indices[:, 1])
                # 获取最大纵坐标值
                max_y_value = prompt_point_indices[max_y_index, 1].item()+5
                max_y = max_y_value if max_y_value <= 512 else 512
                # 获取最小纵坐标值的索引
                min_y_index = torch.argmin(prompt_point_indices[:, 1])
                # 获取最小纵坐标值
                min_y_value = prompt_point_indices[min_y_index, 1].item()-5
                min_y = min_y_value if min_y_value >= 0 else 0
                # 获取最大横坐标值的索引
                max_x_index = torch.argmax(prompt_point_indices[:, 0])
                # 获取最大横坐标值
                max_x_value = prompt_point_indices[max_x_index, 0].item()+5
                max_x = max_x_value if max_x_value <= 512 else 512
                # 获取最小横坐标值的索引
                min_x_index = torch.argmin(prompt_point_indices[:, 0])
                # 获取最小横坐标值
                min_x_value = prompt_point_indices[min_x_index, 0].item()-5
                min_x = min_x_value if min_x_value >= 0 else 0
                # 设置为可输入的格式
                temp_box = torch.tensor([min_y, min_x, max_y, max_x])
                temp_box = torch.unsqueeze(temp_box, 0)
                prompt_box.append(temp_box)

            prompt_box = torch.stack(prompt_box, dim=0).to(device)
            prompt_points = torch.stack(prompt_points, dim=0).to(device)
            point_labels = torch.ones((batch_size, num_points)).to(device)
        else:                                                                  # 直接用所输入的prompt
            assert prompt_box is not None, "Either mask_label or prompt_input must be provided."
        # foreground ones, background zeros
        def prompt_encoder_mode1():
            return sam.prompt_encoder(points=(apply_coords(prompt_points,original_size, sam=sam), point_labels), boxes=(apply_boxes(boxes=prompt_box, original_size=original_size, sam=sam)),masks=None)

        def prompt_encoder_mode2():
            return sam.prompt_encoder(points=None, boxes=(apply_boxes(boxes=prompt_box, original_size=original_size, sam=sam)), masks=None)

        def prompt_encoder_mode3():
            return sam.prompt_encoder(points=(apply_coords(prompt_points,original_size, sam=sam), point_labels), boxes=None, masks=None)

        # 生成一个在[1, 3]范围内的随机整数，表示选择的模式
        selected_mode = torch.randint(1, 7, (1,)).item()
        # 根据选择的模式执行相应的函数
        if selected_mode == 1:
            sparse_embeddings, dense_embeddings = prompt_encoder_mode1()
        elif selected_mode == 2:
            sparse_embeddings, dense_embeddings = prompt_encoder_mode2()
        else:
            sparse_embeddings, dense_embeddings = prompt_encoder_mode3()
        # sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        #     # (N,n,2) n=number of points in a single img
        #     points=(apply_coords(prompt_points,original_size, sam=sam), point_labels),
        #     boxes=(apply_boxes(boxes=prompt_box, original_size=original_size, sam=sam)),
        #     masks=None)
        # *********************************************************************************************************

    # 3. Mask Decoder Forward
    # *************************************************************************************************************
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,)  # if False,
    masks = sam.postprocess_masks(low_res_masks, input_size, original_size)
    # *************************************************************************************************************

    # binarize
    # *************************************************************************************************************
    if return_logits:
        masks = masks > sam.mask_threshold
    # *************************************************************************************************************

    # cast to numpy
    # *************************************************************************************************************
    if numpy:
        masks = masks.detach().cpu().numpy()

    if return_prompt:
        return masks, iou_predictions, low_res_masks
    else:
        return masks, iou_predictions, low_res_masks
    # *************************************************************************************************************