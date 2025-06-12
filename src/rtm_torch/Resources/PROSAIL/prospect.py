# -*- coding: utf-8 -*-
'''

prospect.py executes the Prospect-Model
This routine contains all implemented versions of Prospect
If the code is changed, please make sure to pick the correct (or all) model(s)

Féret J.B., Berger K., de Boissieu F., Malenovský Z. (2020):
PROSPECT-PRO for estimating content of nitrogen-containing leaf proteins and other carbon-based constituents

Féret J.B., Gitelson A.A., Noble S.D., & Jacquemoud S. (2017):
PROSPECT-D: towards modeling leaf optical properties through a complete lifecycle,
Remote Sensing of Environment, 193:204-215. Article.

Féret J.B., François C., Asner G.P., Gitelson A.A., Martin R.E., Bidel L.P.R.,
Ustin S.L., le Maire G., & Jacquemoud S. (2008):
PROSPECT-4 and 5: advances in the leaf optical properties model separating photosynthetic pigments.
Remote Sensing of Environment, 112, 3030-3043. Article.

Jacquemoud S., & Baret F. (1990): PROSPECT: a model of leaf optical properties spectra. Remote Sensing of Environment,
34, 75-91. Article.
'''

import torch
from rtm_torch.Resources.PROSAIL.dataSpec import *
from rtm_torch.Resources.special import exp1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Prospect:

    nlambd = len(lambd)

    def prospect_Pro(self, N, Cab, Car, Anth, Cp, Cbc, Cbrown, Cw):  # Does not contain Cm

        n = Ppro_refractive
        k = (torch.outer(Cab, Ppro_k_Cab) + torch.outer(Car, Ppro_k_Ccx) + torch.outer(Anth, Ppro_k_Canth) +
             # torch.outer(Cm, Ppro_k_Cm)
             torch.outer(Cbrown, Ppro_k_Cbrown) + torch.outer(Cw, Ppro_k_Cw) +
             torch.outer(Cp, Ppro_k_Cp) + torch.outer(Cbc, Ppro_k_Cbc)) / N.unsqueeze(-1)

        ind_k0_row, ind_k0_col = torch.where(k == 0)  # Vectorize = 2D

        if len(ind_k0_row) > 0:
            k[ind_k0_row, ind_k0_col] = torch.finfo(float).eps
        # convert to numpy because exp1 doesn't exist in PyTorch
        trans = (1 - k) * torch.exp(-k) + (k ** 2) * exp1(k)
        trans2 = trans ** 2

        # t12, tav90n are calculated once and are listed in dataSpec
        # t12 is tav(40, n); tav90n is tav(90, n)
        t21 = Ppro_tav90n / (n ** 2)
        r12 = 1 - Ppro_t12
        r21 = 1 - t21
        r21_2 = r21 ** 2
        x = Ppro_t12 / Ppro_tav90n
        y = x * (Ppro_tav90n - 1) + 1 - Ppro_t12

        # reflectance and transmittance of the elementary layer N = 1
        ra = r12 + ((Ppro_t12 * t21 * r21) * trans2) / (1 - (r21_2) * (trans2))
        ta = ((Ppro_t12 * t21) * trans) / (1 - (r21_2) * (trans2))
        r90 = (ra - y) / x
        t90 = ta / x

        # reflectance and transmittance of N layers
        t90_2 = t90 ** 2
        r90_2 = r90 ** 2

        delta = torch.sqrt((t90_2 - r90_2 - 1) ** 2 - 4 * r90_2)
        beta = (1 + r90_2 - t90_2 - delta) / (2 * r90)
        va = (1 + r90_2 - t90_2 + delta) / (2 * r90)

        vb = torch.zeros((Cab.shape[0], self.nlambd)).to(device)

        ind_vb_le_row, ind_vb_le_col = torch.where(va * (beta - r90) <= 1e-14)
        ind_vb_gt_row, ind_vb_gt_col = torch.where(va * (beta - r90) > 1e-14)
        vb[ind_vb_le_row, ind_vb_le_col] = torch.sqrt(beta[ind_vb_le_row, ind_vb_le_col] *
                                                      (va[ind_vb_le_row, ind_vb_le_col] - r90[
                                                          ind_vb_le_row, ind_vb_le_col]) / 1e-14)
        vb[ind_vb_gt_row, ind_vb_gt_col] = torch.sqrt(beta[ind_vb_gt_row, ind_vb_gt_col] *
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] - r90[
                                                          ind_vb_gt_row, ind_vb_gt_col]) /
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] *
                                                       (beta[ind_vb_gt_row, ind_vb_gt_col] - r90[
                                                           ind_vb_gt_row, ind_vb_gt_col])))

        vbNN = vb ** ((N - 1).unsqueeze(-1))
        vbNNinv = 1 / vbNN
        vainv = 1 / va
        s1 = ta * t90 * (vbNN - vbNNinv)
        s2 = ta * (va - vainv)
        s3 = va * vbNN - vainv * vbNNinv - r90 * (vbNN - vbNNinv)

        RN = ra + s1 / s3
        TN = s2 / s3
        LRT = torch.zeros((Cab.shape[0], self.nlambd, 3))
        LRT[:, :, 0] = lambd.unsqueeze(0)
        LRT[:, :, 1] = RN
        LRT[:, :, 2] = TN

        return LRT

    def prospect_D(self, N, Cab, Car, Anth, Cbrown, Cw, Cm):
        n = PD_refractive
        # NOTE if N is a zero tensor, the result of k is inf
        k = (torch.outer(Cab, PD_k_Cab) + torch.outer(Car, PD_k_Car) + torch.outer(Anth, PD_k_Anth) +
             torch.outer(Cbrown, PD_k_Brown) + torch.outer(Cw, PD_k_Cw) + torch.outer(Cm, PD_k_Cm)) / N.unsqueeze(-1)

        ind_k0_row, ind_k0_col = torch.where(k == 0)  # Vectorize = 2D

        if len(ind_k0_row) > 0:
            k[ind_k0_row, ind_k0_col] = torch.finfo(float).eps
        trans = (1 - k) * torch.exp(-k) + (k ** 2) * exp1(k)
        trans2 = trans ** 2

        # t12, tav90n are calculated once and are listet in dataSpec
        # t12 is tav(40, n); tav90n is tav(90, n)
        t21 = PD_tav90n / (n ** 2)
        r12 = 1 - PD_t12
        r21 = 1 - t21
        r21_2 = r21 ** 2
        x = PD_t12 / PD_tav90n
        y = x * (PD_tav90n - 1) + 1 - PD_t12

        # reflectance and transmittance of the elementary layer N = 1
        ra = r12 + ((PD_t12 * t21 * r21) * trans2) / (1 - (r21_2) * (trans2))
        ta = ((PD_t12 * t21) * trans) / (1 - (r21_2) * (trans2))
        r90 = (ra - y) / x
        t90 = ta / x

        # reflectance and transmittance of N layers
        t90_2 = t90 ** 2
        r90_2 = r90 ** 2

        delta = torch.sqrt((t90_2 - r90_2 - 1) ** 2 - 4 * r90_2)
        beta = (1 + r90_2 - t90_2 - delta) / (2 * r90)
        va = (1 + r90_2 - t90_2 + delta) / (2 * r90)

        vb = torch.zeros((Cab.shape[0], self.nlambd)).to(device)

        ind_vb_le_row, ind_vb_le_col = torch.where(va * (beta-r90) <= 1e-14)
        ind_vb_gt_row, ind_vb_gt_col = torch.where(va * (beta-r90) > 1e-14)
        vb[ind_vb_le_row, ind_vb_le_col] = torch.sqrt(beta[ind_vb_le_row, ind_vb_le_col] *
                                                      (va[ind_vb_le_row, ind_vb_le_col] - r90[ind_vb_le_row, ind_vb_le_col]) / 1e-14)
        vb[ind_vb_gt_row, ind_vb_gt_col] = torch.sqrt(beta[ind_vb_gt_row, ind_vb_gt_col] *
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col]) /
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] *
                                                       (beta[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col])))

        vbNN = vb ** ((N - 1).unsqueeze(-1))
        vbNNinv = 1 / vbNN
        vainv = 1 / va
        s1 = ta * t90 * (vbNN - vbNNinv)
        s2 = ta * (va - vainv)
        s3 = va * vbNN - vainv * vbNNinv - r90 * (vbNN - vbNNinv)

        RN = ra + s1 / s3
        TN = s2 / s3
        LRT = torch.zeros((Cab.shape[0], self.nlambd, 3))
        LRT[:, :, 0] = lambd.unsqueeze(0)
        LRT[:, :, 1] = RN
        LRT[:, :, 2] = TN

        return LRT

    def prospect_5(self, N, Cab, Car, Cw, Cm):
        n = P5_refractive
        k = (torch.outer(Cab, P5_k_Cab) + torch.outer(Car, P5_k_Car) +
             torch.outer(Cw, P5_k_Cw) + torch.outer(Cm, P5_k_Cm)) / N.unsqueeze(-1)

        ind_k0_row, ind_k0_col = torch.where(k == 0)  # Vectorize = 2D

        if len(ind_k0_row) > 0:
            k[ind_k0_row, ind_k0_col] = torch.finfo(float).eps
        trans = (1 - k) * torch.exp(-k) + (k ** 2) * exp1(k)
        trans2 = trans ** 2

        t21 = P5_tav90n / (n ** 2)
        r12 = 1 - P5_t12
        r21 = 1 - t21
        r21_2 = r21 ** 2
        x = P5_t12 / P5_tav90n
        y = x * (P5_tav90n - 1) + 1 - P5_t12

        ra = r12 + ((P5_t12 * t21 * r21) * trans2) / (1 - (r21_2) * (trans2))
        ta = ((P5_t12 * t21) * trans) / (1 - (r21_2) * (trans2))
        r90 = (ra - y) / x
        t90 = ta / x

        t90_2 = t90 ** 2
        r90_2 = r90 ** 2

        delta = torch.sqrt((t90_2 - r90_2 - 1) ** 2 - 4 * r90_2)
        beta = (1 + r90_2 - t90_2 - delta) / (2 * r90)
        va = (1 + r90_2 - t90_2 + delta) / (2 * r90)

        vb = torch.zeros((Cab.shape[0], self.nlambd)).to(device)

        ind_vb_le_row, ind_vb_le_col = torch.where(va * (beta - r90) <= 1e-14)
        ind_vb_gt_row, ind_vb_gt_col = torch.where(va * (beta - r90) > 1e-14)
        vb[ind_vb_le_row, ind_vb_le_col] = torch.sqrt(beta[ind_vb_le_row, ind_vb_le_col] *
                                                      (va[ind_vb_le_row, ind_vb_le_col] - r90[ind_vb_le_row, ind_vb_le_col]) / 1e-14)
        vb[ind_vb_gt_row, ind_vb_gt_col] = torch.sqrt(beta[ind_vb_gt_row, ind_vb_gt_col] *
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col]) /
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] *
                                                       (beta[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col])))

        vbNN = vb ** ((N - 1).unsqueeze(-1))
        vbNNinv = 1 / vbNN
        vainv = 1 / va
        s1 = ta * t90 * (vbNN - vbNNinv)
        s2 = ta * (va - vainv)
        s3 = va * vbNN - vainv * vbNNinv - r90 * (vbNN - vbNNinv)

        RN = ra + s1 / s3
        TN = s2 / s3
        LRT = torch.zeros((Cab.shape[0], self.nlambd, 3))
        LRT[:, :, 0] = lambd.unsqueeze(0)
        LRT[:, :, 1] = RN
        LRT[:, :, 2] = TN

        return LRT

    def prospect_5B(self, N, Cab, Car, Cbrown, Cw, Cm):

        n = P5_refractive
        k = (torch.outer(Cab, P5_k_Cab) + torch.outer(Car, P5_k_Car) +
             torch.outer(Cbrown, P5_k_Brown) + torch.outer(Cw, P5_k_Cw) + torch.outer(Cm, P5_k_Cm)) / N.unsqueeze(-1)

        ind_k0_row, ind_k0_col = torch.where(k == 0)  # Vectorize = 2D

        if len(ind_k0_row) > 0:
            k[ind_k0_row, ind_k0_col] = torch.finfo(float).eps
        # Using torch.exp(-k) / k as approximation of exp1(k)
        trans = (1 - k) * torch.exp(-k) + (k ** 2) * exp1(k)
        trans2 = trans ** 2

        # t12, tav90n are calculated once and are listed in dataSpec
        # t12 is tav(40, n); tav90n is tav(90, n)
        t21 = P5_tav90n / (n ** 2)
        r12 = 1 - P5_t12
        r21 = 1 - t21
        r21_2 = r21 ** 2
        x = P5_t12 / P5_tav90n
        y = x * (P5_tav90n - 1) + 1 - P5_t12

        # reflectance and transmittance of the elementary layer N = 1
        ra = r12 + ((P5_t12 * t21 * r21) * trans2) / (1 - (r21_2) * (trans2))
        ta = ((P5_t12 * t21) * trans) / (1 - (r21_2) * (trans2))
        r90 = (ra - y) / x
        t90 = ta / x

        # reflectance and transmittance of N layers
        t90_2 = t90 ** 2
        r90_2 = r90 ** 2

        delta = torch.sqrt((t90_2 - r90_2 - 1) ** 2 - 4 * r90_2)
        beta = (1 + r90_2 - t90_2 - delta) / (2 * r90)
        va = (1 + r90_2 - t90_2 + delta) / (2 * r90)

        vb = torch.zeros((Cab.shape[0], self.nlambd)).to(device)

        ind_vb_le_row, ind_vb_le_col = torch.where(va * (beta - r90) <= 1e-14)
        ind_vb_gt_row, ind_vb_gt_col = torch.where(va * (beta - r90) > 1e-14)
        vb[ind_vb_le_row, ind_vb_le_col] = torch.sqrt(beta[ind_vb_le_row, ind_vb_le_col] *
                                                      (va[ind_vb_le_row, ind_vb_le_col] - r90[ind_vb_le_row, ind_vb_le_col]) / (1e-14))
        vb[ind_vb_gt_row, ind_vb_gt_col] = torch.sqrt(beta[ind_vb_gt_row, ind_vb_gt_col] *
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col]) /
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] * (beta[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col])))

        vbNN = vb ** ((N - 1).unsqueeze(-1))
        vbNNinv = 1 / vbNN
        vainv = 1 / va
        s1 = ta * t90 * (vbNN - vbNNinv)
        s2 = ta * (va - vainv)
        s3 = va * vbNN - vainv * vbNNinv - r90 * (vbNN - vbNNinv)

        RN = ra + s1 / s3
        TN = s2 / s3
        LRT = torch.zeros((Cab.shape[0], self.nlambd, 3))
        LRT[:, :, 0] = lambd.unsqueeze(0)
        LRT[:, :, 1] = RN
        LRT[:, :, 2] = TN

        return LRT

    def prospect_4(self, N, Cab, Cw, Cm):
        n = P4_refractive
        k = (torch.outer(Cab, P4_k_Cab) + torch.outer(Cw, P4_k_Cw) +
             torch.outer(Cm, P4_k_Cm)) / N.unsqueeze(-1)

        ind_k0_row, ind_k0_col = torch.where(k == 0)  # Vectorize = 2D

        if len(ind_k0_row) > 0:
            k[ind_k0_row, ind_k0_col] = torch.finfo(float).eps
        trans = (1 - k) * torch.exp(-k) + (k ** 2) * exp1(k)
        trans2 = trans ** 2

        # t12, tav90n are calculated once and are listed in dataSpec
        # t12 is tav(40, n); tav90n is tav(90, n)
        t21 = P4_tav90n / (n ** 2)
        r12 = 1 - P4_t12
        r21 = 1 - t21
        r21_2 = r21 ** 2
        x = P4_t12 / P4_tav90n
        y = x * (P4_tav90n - 1) + 1 - P4_t12

        # reflectance and transmittance of the elementary layer N = 1
        ra = r12 + ((P4_t12 * t21 * r21) * trans2) / (1 - (r21_2) * (trans2))
        ta = ((P4_t12 * t21) * trans) / (1 - (r21_2) * (trans2))
        r90 = (ra - y) / x
        t90 = ta / x

        # reflectance and transmittance of N layers
        t90_2 = t90 ** 2
        r90_2 = r90 ** 2

        delta = torch.sqrt((t90_2 - r90_2 - 1) ** 2 - 4 * r90_2)
        beta = (1 + r90_2 - t90_2 - delta) / (2 * r90)
        va = (1 + r90_2 - t90_2 + delta) / (2 * r90)

        vb = torch.zeros((Cab.shape[0], self.nlambd))

        ind_vb_le_row, ind_vb_le_col = torch.where(va * (beta - r90) <= 1e-14)
        ind_vb_gt_row, ind_vb_gt_col = torch.where(va * (beta - r90) > 1e-14)
        vb[ind_vb_le_row, ind_vb_le_col] = torch.sqrt(beta[ind_vb_le_row, ind_vb_le_col] *
                                                      (va[ind_vb_le_row, ind_vb_le_col] - r90[ind_vb_le_row, ind_vb_le_col]) / (1e-14))
        vb[ind_vb_gt_row, ind_vb_gt_col] = torch.sqrt(beta[ind_vb_gt_row, ind_vb_gt_col] *
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col]) /
                                                      (va[ind_vb_gt_row, ind_vb_gt_col] *
                                                       (beta[ind_vb_gt_row, ind_vb_gt_col] - r90[ind_vb_gt_row, ind_vb_gt_col])))

        vbNN = vb ** ((N - 1).unsqueeze(-1))
        vbNNinv = 1 / vbNN
        vainv = 1 / va
        s1 = ta * t90 * (vbNN - vbNNinv)
        s2 = ta * (va - vainv)
        s3 = va * vbNN - vainv * vbNNinv - r90 * (vbNN - vbNNinv)

        RN = ra + s1 / s3
        TN = s2 / s3
        LRT = torch.zeros((Cab.shape[0], self.nlambd, 3))
        LRT[:, :, 0] = lambd.unsqueeze(0)
        LRT[:, :, 1] = RN
        LRT[:, :, 2] = TN

        return LRT
