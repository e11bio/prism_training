import torch


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
