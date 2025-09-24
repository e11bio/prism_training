import torch


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
