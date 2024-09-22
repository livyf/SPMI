import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class v4Loss(nn.Module):
    def __init__(self, predicted_score_cls):
        super().__init__()
        self.predicted_score_cls1 = predicted_score_cls

    def update_weight_byclsout1(self, cls_predicted_score, batch_index, batch_partial_Y, args):
        with torch.no_grad():
            
            revisedY_raw = batch_partial_Y.clone()
            revisedY_raw = revisedY_raw * cls_predicted_score
            revisedY_raw = revisedY_raw / revisedY_raw.sum(dim=1).repeat(args.num_class, 1).transpose(0, 1)

            cls_pseudo_label = revisedY_raw.detach()
            
            self.predicted_score_cls1[batch_index, :] = cls_pseudo_label

    def forward(self, cls_out_w, index):

        soft_positive_label_target1 = self.predicted_score_cls1[index, :].clone().detach()

        cls_loss_w = -torch.mean(torch.sum(soft_positive_label_target1 * torch.log(cls_out_w), dim=1))

        total_loss = cls_loss_w

        return total_loss


# complementary loss (Learning from complementary labels. NIPS'17 https://github.com/takashiishida/comp)
# pc
def pc_loss(f, K, labels):
    """
    :param f: output of model
    :param K: number of classes
    :param labels: complementary label
    :return: loss
    """
    fbar = f.gather(1, labels).repeat(1, K)
    loss_matrix = torch.sigmoid( -1. * (f - fbar)) # multiply -1 for "complementary"
    M1, M2 = K*(K-1)/2, K-1
    pc_loss = torch.sum(loss_matrix)*(K-1)/len(labels) - M1 + M2
    return pc_loss

# ga
def non_negative_loss(f, K, labels, ccp, beta):
    ccp = torch.from_numpy(ccp).float().cuda()
    neglog = -F.log_softmax(f, dim=1)
    loss_vector = torch.zeros(K, requires_grad=True).cuda()
    temp_loss_vector = torch.zeros(K).cuda()
    for k in range(K):
        idx = labels == k
        if torch.sum(idx).item() > 0:
            idxs = idx.byte().view(-1,1).repeat(1,K)
            neglog_k = torch.masked_select(neglog, idxs).view(-1,K)
            temp_loss_vector[k] = -(K-1) * ccp[k] * torch.mean(neglog_k, dim=0)[k]  # average of k-th class loss for k-th comp class samples
            loss_vector = loss_vector + torch.mul(ccp[k], torch.mean(neglog_k, dim=0))  # only k-th in the summation of the second term inside max
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < K:
        count = np.append(count, 0) # when largest label is below K, bincount will not take care of them
    loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).cuda()-beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss, torch.mul(torch.from_numpy(count).float().cuda(), loss_vector)


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
        