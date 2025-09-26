import torch

from itertools import combinations

import torch

import wandb

import time


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        t: int = 1,
        a: int = 1,
        regularize: bool = False,
        background_relabel: bool = False,
        dynamic_balancing: bool = False,
        uniform_emb: bool = False,
        batchwise: bool = True,
        prefix: str = "",
    ):
        super().__init__()

        # todo - add docstrings

        self.t = t
        self.a = a
        self.regularize = regularize
        self.background_relabel = background_relabel
        self.dynamic_balancing = dynamic_balancing
        self.uniform_emb = uniform_emb
        self.batchwise = batchwise
        self.prefix = prefix

    def forward(self, prediction, gt):
        """
        given ground truth and prediction, compute loss that minimizes
        the variance of the prediction for each label in the ground truth,
        and maximizes the distance between the center_means of the prediction within
        each ground truth label
        """
        t1 = time.time()
        b, c, z, y, x = prediction.shape
        gt = gt.view(b, z, y, x)
        if self.background_relabel:
            gt += 1

        if not self.batchwise:
            intra_similarities = []
            inter_similarities = []
            for sample in range(b):
                (
                    intra_similarities_sample,
                    inter_similarities_sample,
                ) = self.compute_similarities(prediction[sample], gt[sample])
                intra_similarities.append(intra_similarities_sample)
                inter_similarities.append(inter_similarities_sample)
            intra_similarities = torch.cat(intra_similarities, dim=0)
            inter_similarities = torch.cat(inter_similarities, dim=0)
        else:
            intra_similarities, inter_similarities = self.compute_similarities(
                prediction.transpose(0, 1), gt
            )

        intra_similarity = torch.mean(intra_similarities)

        # can we get rid of this line?
        # intra_loss = torch.mean((1 - intra_similarities) ** 2)

        intra_loss = -intra_similarity

        if self.regularize:
            normed_pred = torch.nn.functional.normalize(prediction, p=2, dim=1)
            reg_loss = (
                torch.nn.MSELoss()(prediction, normed_pred) if self.regularize else None
            )

        if len(inter_similarities) > 0:
            inter_similarity = torch.mean(inter_similarities)
            clipped_inter_similarity = torch.mean(torch.clamp(inter_similarities, 0, 1))

            gaussian_potential_similarity = (
                torch.log(
                    torch.mean(torch.exp(2 * self.t * inter_similarities - 2 * self.t))
                )
                / (2 * self.t)
            ) + 1

            inter_loss = (
                gaussian_potential_similarity
                if self.uniform_emb
                else clipped_inter_similarity
            )
            balance_term = 1 - inter_loss if self.dynamic_balancing else 1

            try:
                wandb.log(
                    {
                        f"{self.prefix}inter_similarity": inter_similarity,
                        f"{self.prefix}clipped_inter_similarity": clipped_inter_similarity,
                        f"{self.prefix}gaussian_potential_similarity": gaussian_potential_similarity,
                        f"{self.prefix}balance_term": (
                            balance_term if self.dynamic_balancing else None
                        ),
                    },
                    commit=False,
                )
            except wandb.errors.Error as e:
                print(
                    e,
                    (
                        f"{self.prefix}inter_similarity {inter_similarity.item():.3f}\n"
                        f"{self.prefix}clipped_inter_similarity "
                        f"{clipped_inter_similarity.item():.3f}\n"
                        f"{self.prefix}gaussian_potential_similarity "
                        f"{gaussian_potential_similarity.item():.3f}"
                    ),
                )
        else:
            balance_term = torch.tensor(1, device=prediction.device)
            inter_loss = torch.tensor(0, device=prediction.device)

        try:
            wandb.log(
                {
                    f"{self.prefix}intra_similarity": intra_similarity,
                    f"{self.prefix}reg_loss": reg_loss if self.regularize else None,
                },
                commit=False,
            )
        except wandb.errors.Error as e:
            print(
                e,
                (
                    f"{self.prefix}intra_similarity {intra_similarity.item():.3f}\n"
                    f"{self.prefix}reg_loss "
                    f"{(reg_loss.item() if self.regularize else 0.0):.3f}\n"
                ),
            )

        print(f"Contrastive loss computed in {time.time() - t1:.3f} seconds")

        return (
            inter_loss
            + balance_term * intra_loss
            + (reg_loss if self.regularize else 0)
        )

    def compute_similarities(self, prediction, gt):
        # Compute the variance of the prediction for each label in the gt
        c, *_ = prediction.shape
        intra_similarities = []
        unique_labels = list(torch.unique(gt))
        center_means = []
        for label in unique_labels:
            if label == 0:
                continue
            mean_vec = []
            mask = gt == label
            masked_preds = []
            for i in range(c):
                masked_pred = prediction[i][mask]
                masked_preds.append(masked_pred)

                mean = torch.mean(masked_pred)
                mean_vec.append(mean)
            masked_pred = torch.stack(masked_preds)
            mean_vec = torch.stack(mean_vec)

            if label > 0 and len(masked_pred) > 1:
                dot_prod = torch.nn.CosineSimilarity(dim=0)(
                    masked_pred, mean_vec.view(c, 1)
                )
                intra_similarities.append(torch.mean(dot_prod))

            center_means.append(mean_vec)

        intra_similarities = torch.stack(intra_similarities)

        # Compute the distance between the center_means of the prediction within
        # each ground truth label
        if len(center_means) < 2:
            return intra_similarities, torch.empty(0, device=prediction.device)
        inter_similarities = []
        for center_a, center_b in combinations(center_means, 2):
            similarity = torch.nn.CosineSimilarity(dim=0)(center_a, center_b)
            inter_similarities.append(similarity)
        inter_similarities = torch.stack(inter_similarities)

        return intra_similarities, inter_similarities


class WeightedBCELossMultiChannel(torch.nn.BCELoss):
    def __init__(self, reduction="mean"):
        """
        Custom BCELoss for 2-channel output (mask + contour) with pixel-wise weighting.

        weight_mask: Tensor of same shape as input, defining per-pixel weights
            for the mask.
        weight_contour: Tensor of same shape as input, defining per-pixel
            weights for the contour.
        mask_weight: Global weight for the mask loss.
        contour_weight: Global weight for the contour loss.
        reduction: 'mean' or 'sum' (default: 'mean').
        """

        super().__init__(reduction="none")  # We handle reduction manually

        self.reduction = reduction

    def forward(self, input, target, weight):
        """
        Compute BCE loss separately for mask and contour, applying per-pixel weights.
        """

        # Split the target into mask and contour channels
        target_mask = target[0, :, :, :]  # First channel = mask
        target_contour = target[1, :, :, :]  # Second channel = contour

        pred_mask = input[0, :, :, :]  # First channel = mask
        pred_contour = input[1, :, :, :]  # Second channel = contour

        # Compute BCE loss separately for mask and contour
        loss_mask = super().forward(pred_mask, target_mask)
        loss_contour = super().forward(pred_contour, target_contour)

        weight_contour = target_contour
        weight_mask = weight

        loss_contour = (1 + 12.0 * weight_contour) * loss_contour
        loss_mask = (1 + 2.0 * weight_mask) * loss_mask

        # Handle reduction manually
        if self.reduction == "mean":
            return loss_mask.mean() + loss_contour.mean()

        elif self.reduction == "sum":
            return loss_mask.sum() + loss_contour.sum()

        else:
            return loss_mask + loss_contour  # 'none' returns element-wise loss


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

        # todo: add docstrings

    def _calc_loss(self, pred, target, weights):
        scaled = weights * (pred - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        prediction,
        target,
        weights,
        aux_prediction=None,
        aux_target=None,
        aux_weights=None,
    ):
        loss = self._calc_loss(prediction, target, weights)

        if aux_prediction is not None:
            assert aux_target is not None and aux_weights is not None, (
                "Must provide auxiliary target and weights "
                "if providing auxiliary predictions"
            )

            aux_loss = self._calc_loss(aux_prediction, aux_target, aux_weights)

            loss += aux_loss

        return loss
