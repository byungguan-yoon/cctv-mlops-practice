import torch
import wandb

from pytorch_lightning import Callback


# Custom Callback
class ImagePredictionLogger(Callback):
    def __init__(self, val_sample_image, val_sample_gt, mode, val_sample_meta=None):
        super().__init__()
        self.val_imgs = val_sample_image
        self.val_gt = val_sample_gt
        self.val_meta = val_sample_meta
        self.mode = mode

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.float().to(device=pl_module.device)
        val_labels = self.val_gt.to(device=pl_module.device)

        # Get model prediction
        if not self.val_meta is None:
            val_meta = self.val_meta.float().to(device=pl_module.device)
            val_preds = pl_module((val_imgs, val_meta))
        else:
            val_preds = pl_module(val_imgs)

        if self.mode in ["reg", "cls"]:
            val_preds = torch.argmax(val_preds, dim=1)

        log_list = list()
        for val_img, val_pred, val_label in zip(val_imgs, val_preds, val_labels):
            if self.mode == "seg":
                show_log = self.wandb_seg_image(
                    val_img.detach().cpu().permute(1, 2, 0).numpy(),
                    val_pred.detach().cpu().numpy(),
                    val_label.detach().cpu().numpy(),
                )
            elif self.mode in ["reg", "cls"]:
                show_log = self.wandb_reg_image(
                    val_img.detach().cpu().permute(1, 2, 0).numpy(),
                    val_pred.detach().cpu().numpy(),
                    val_label.detach().cpu().numpy(),
                )
            log_list.append(show_log)

        # Log the images as wandb Image
        trainer.logger.experiment.log({"examples": log_list}, commit=False)

    def wandb_reg_image(self, image, pred, gt):
        return wandb.Image(image, caption=f"pred : {pred.item()}, gt : {gt}")

    def wandb_seg_image(self, image, pred_mask, true_mask):

        return wandb.Image(
            image,
            masks={
                "prediction": {"mask_data": pred_mask, "class_labels": self.labels()},
                "ground truth": {"mask_data": true_mask, "class_labels": self.labels()},
            },
        )

    def labels(self):
        l = {}
        for i, label in enumerate(self.category_names):
            l[i] = label
        return l
