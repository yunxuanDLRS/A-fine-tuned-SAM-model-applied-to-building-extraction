from segment_anything import sam_model_registry
from utils.functions import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.sam_loss import SamLoss
from utils.sam_dataset_3 import SamDataset_train, SamDataset_test
from utils.sam_forward import SamForward
import time
import numpy as np
np.set_printoptions(threshold=np.inf)

def main():
    """
    SAM model maps: (image, prompt) -> (mask)
    The model is prompted with same random single point from mask label.
    Binary accuracy of thresholded mask predictions is monitored, and the decoder model is saved when the highest validation accuracy is achieved.
    """
    global sam, device
    # Load SAM model
    checkpoint = 'model/sam_vit_b_01ec64.pth'
    # device = 'cuda'
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry['vit_b'](
        checkpoint=checkpoint).to(device)  # ViT-Huge
    sam.mask_decoder.load_state_dict(torch.load('output_test2/finetuned_decoder_epoch95_batch1900_score0.8902.pth'))

    # Initial configs
    # todo: layerwise learning rate decay of 0.8 not properly applied
    # todo: drop-path with rate of 0.4
    # todo: decreasing lr with factor of 10 at epoch 60000, 86666...not considered
    sam.image_encoder.eval()  # ViT-H image encoder
    sam.prompt_encoder.eval()  # SAM prompt encoder
    sam.mask_decoder.train()  # Lightweight mask decoder
    optimizer = torch.optim.AdamW([{'params': sam.mask_decoder.parameters(
    ), 'lr': 1e-5, 'betas': (0.9, 0.999), 'weight_decay': 0.1}])  # LR= SAM final training lr(8e-6)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = SamLoss()

    # Load dataset
    # fixme: data loading seems weird
    train_dataloader = SamDataset_train()
    # Batch size more than 1 cause error (due to multi-prompt)
    # https://github.com/facebookresearch/segment-anything/issues/277
    train_dataloader = DataLoader(train_dataloader, batch_size=1, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)

    test_dataloader = SamDataset_test()
    test_dataloader = DataLoader(test_dataloader, batch_size=1, shuffle=True, pin_memory=True, num_workers=1, persistent_workers=True)
    # Training Loop
    steps = 0
    steps_max = 35  #gradient accumulation steps,较小的梯度累积步数可能更适合显存受限的情况，而较大的梯度累积步数可能对训练稳定性更有帮助。
    epoch_step = len(train_dataloader) / steps_max
    scores_train = []
    loss_train = []
    scores_test = []
    score_train = 0
    batched_loss_train = 0
    batch_count = 0
    for epoch in range(100):
        # training batch loop
        sam.mask_decoder.train()
        for idx, batch in enumerate(train_dataloader):
            img_label, mask_label = batch # mask_label.shape torch.Size([1, 512, 512]) img_label.shape torch.Size([1, 512, 512, 3])
            img_label = img_label.to(device)
            mask_label = mask_label.to(device)
            # forward
            masks, iou_predictions, low_res_masks = SamForward(sam,
                                                               img_label, mask_label, device=device)  # take only coarse mask
            # compute loss and grad
            # print(type(mask_label))
            # print(type(masks[:, 0, ...]))
            loss = loss_fn(masks[:, 0, ...], mask_label, iou_predictions)
            loss /= steps_max
            # 计算损失函数对模型参数的梯度，从而进行反向传播
            loss.backward()
            # 使用.item()方法将其转换为 Python 数值
            batched_loss_train += loss.item()
            steps += 1
            # evaluate scores with logits
            # 内部的操作不需要计算梯度
            with torch.no_grad():
                mask_label_logits = mask_label.type(torch.bool)
                mask_pred_logits = masks > sam.mask_threshold
                score_train += SamLoss().iou_logits(mask_pred_logits, mask_label_logits).item()/steps_max
            # update acuumulated grads
            # 每steps_max步计算一次指数【总共就会计算data.num/step_max次】
            if steps == steps_max:
                print(
                    f"Epoch {epoch+1}, stepping at batch {idx+1}/{len(train_dataloader)}, mIoU score={score_train:.4f}, loss={batched_loss_train:.4f}")
                # record score log
                # 计算参考指数。（最终会计算【data.num/step_max*迭代次数】次）
                scores_train.append(score_train)
                loss_train.append(batched_loss_train)

                # backprop acuumulations 新模型参数，根据梯度和选择的优化算法来进行参数更新。
                optimizer.step()
                # 清除这些参数的梯度，为下一轮训练做准备。防止梯度在多个批次之间累积。
                for p in sam.mask_decoder.parameters():
                    p.grad = None
                batch_count += 1
                # initialize
                steps = 0
                batched_loss_train = 0
                score_train = 0
        # End of every update
        name = f"finetuned_decoder_epoch{epoch+1:02d}_batch{batch_count:04d}_score{float(sum(scores_train[-20:])/20):.4f}"
        # 模型保存
        if epoch % 2 == 0:
            sam.mask_decoder.to('cpu')
            best_decoder_param = deepcopy(sam.mask_decoder.state_dict())
            sam.mask_decoder.to(device)
            torch.save(best_decoder_param, f'output/{name}.pth')

            log_dict = {"scores_train": scores_train,
                        "loss_train": loss_train}
            torch.save(log_dict, f'output/{name}.ptlog')
        print("|| Epoch: %d ||  score avg: %.4f || loss avg: %.4f ||" %((epoch+1), float(sum(scores_train[-20:])/20), float(sum(loss_train[-20:])/20),))

        # test batch loop
        sam.mask_decoder.eval()
        score_test = 0
        steps_test = len(test_dataloader)   # 由测试数据的图像数目确定
        with torch.no_grad():
            for idx_test, batch_test in enumerate(test_dataloader):
                img_label_test, mask_label_test = batch_test
                img_label_test = img_label_test.to(device)
                mask_label_test = mask_label_test.to(device)
                masks_test, iou_predictions_test, low_res_masks_test = SamForward(sam, img_label_test, mask_label_test, device=device)  # take only coarse mask
                mask_label_test_logits = mask_label_test.type(torch.bool)
                mask_pred_test_logits = masks_test > sam.mask_threshold
                score_test += SamLoss().iou_logits(mask_pred_test_logits, mask_label_test_logits).item() / steps_test
            scores_test.append(score_test)
            print("|| Epoch: %d || score_test: %.4f ||" %((epoch+1), score_test,))
        log_dict_test = {'scores_test': scores_test}
        torch.save(log_dict_test, f'output/scores_test.ptlog')

if __name__ == '__main__':
    main()