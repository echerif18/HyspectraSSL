# -*- coding: utf-8 -*-
'''
SAIL.py executes the SAIL model

References:
Verhoef W., Xiao Q., Jia L., & Su Z. (2007):
Unified optical-thermal four-stream radiative transfer theory for homogeneous vegetation canopies.
IEEE Transactions on Geoscience and Remote Sensing, 45, 1808-1822. Article.

Verhoef W., & Bach H. (2003), Simulation of hyperspectral and directional radiance images using coupled biophysical
and atmospheric radiative transfer models. Remote Sensing of Environment, 87, 23-41. Article.

Verhoef W. (1984), Light scattering by leaf layers with application to canopy reflectance modeling: the SAIL model.
Remote Sensing of Environment, 16, 125-141. Article.
'''

import torch
import math
from rtm_torch.Resources.PROSAIL.dataSpec import *
from rtm_torch.Resources.PROSAIL.SAILdata import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


class Sail:
    def __init__(self, tts, tto, psi):
        self.tts = tts
        self.tto = tto
        self.psi = psi

        # Conversions are done in the __init__ to save time (needed as parameters for the complete set of paras)
        self.sintts = torch.sin(tts)
        self.sintto = torch.sin(tto)
        self.costts = torch.cos(tts)
        self.costto = torch.cos(tto)
        self.cospsi = torch.cos(psi)

    def pro4sail(self, rho, tau, LIDF, TypeLIDF, LAI, hspot, psoil, soil,
                 understory=None, skyl=None, inform_trans=None):

        if inform_trans == 'tto':
            self.tts = self.tto
            self.sintts = self.costto
            self.costts = self.costto

        LAI = LAI.unsqueeze(1)  # expand LAI-array to 2D
        costts_costto = self.costts * self.costto
        tantts = torch.tan(self.tts)
        tantto = torch.tan(self.tto)
        dso = torch.sqrt(tantts ** 2 + tantto ** 2 - 2 *
                         tantts * tantto * self.cospsi)

        # Soil Reflectance Properties
        if isinstance(understory, torch.Tensor):
            soil = understory
        # "soil" is not supplied as np.array, but is "None" instead
        elif not isinstance(soil, torch.Tensor):
            # np.outer = outer product (vectorized)
            soil = torch.outer(psoil, Rsoil1) + torch.outer((1-psoil), Rsoil2)

        # Generate Leaf Angle Distribution From Average Leaf Angle (ellipsoidal) or (a, b) parameters
        lidf = self.lidf_calc(LIDF, TypeLIDF)

        # Weighted Sums of LIDF
        litab = torch.cat(
            (torch.arange(5, 85, 10), torch.arange(81, 91, 2)), dim=0).to(device)
        # litab -> 5, 15, 25, 35, 45, ... , 75, 81, 83, ... 89
        litab = litab * (math.pi / 180)

        chi_s, chi_o, frho, ftau = self.volscatt(litab)

        # Extinction coefficients
        ksli = chi_s / self.costts.unsqueeze(1)
        koli = chi_o / self.costto.unsqueeze(1)

        # Area scattering coefficient fractions
        sobli = frho * math.pi / costts_costto.unsqueeze(1)
        sofli = ftau * math.pi / costts_costto.unsqueeze(1)
        bfli = torch.cos(litab) ** 2

        # Angular Differences
        ks = torch.sum(ksli * lidf, dim=1).unsqueeze(1)
        ko = torch.sum(koli * lidf, dim=1).unsqueeze(1)
        bf = torch.sum(bfli[None, :] * lidf, dim=1).unsqueeze(1).to(device)
        sob = torch.sum(sobli * lidf, dim=1).unsqueeze(1)
        sof = torch.sum(sofli * lidf, dim=1).unsqueeze(1)

        # Geometric factors to be used later with reflectance and transmission
        sdb = 0.5 * (ks + bf)
        sdf = 0.5 * (ks - bf)
        dob = 0.5 * (ko + bf)
        dof = 0.5 * (ko - bf)
        ddb = 0.5 * (1 + bf)
        ddf = 0.5 * (1 - bf)

        rho = rho.to(device)
        tau = tau.to(device)

        # Refl and Transm kick in
        sigb = ddb * rho + ddf * tau
        sigf = ddf * rho + ddb * tau
        att = 1.0 - sigf
        m2 = (att + sigb) * (att - sigb)
        m2[m2 < 0] = 0.0
        m = torch.sqrt(m2)

        sb = sdb*rho + sdf*tau
        sf = sdf*rho + sdb*tau
        vb = dob*rho + dof*tau
        vf = dof*rho + dob*tau
        w = sob*rho + sof*tau

        # Include LAI (make sure, LAI is > 0!)
        e1 = torch.exp(-m * LAI)
        e2 = e1 ** 2
        rinf = (att - m) / sigb
        rinf2 = rinf ** 2
        re = rinf * e1
        denom = 1.0 - rinf2 * e2

        J1ks, tss = self.jfunc1(ks, m, LAI)
        J2ks = self.jfunc2(ks, m, LAI)
        J1ko, too = self.jfunc1(ko, m, LAI)
        J2ko = self.jfunc2(ko, m, LAI)

        Ps = (sf + sb * rinf) * J1ks
        Qs = (sf * rinf + sb) * J2ks
        Pv = (vf + vb * rinf) * J1ko
        Qv = (vf * rinf + vb) * J2ko

        rdd = rinf * (1.0 - e2) / denom
        tdd = (1.0 - rinf2) * e1 / denom
        tsd = (Ps - re * Qs) / denom
        tdo = (Pv - re * Qv) / denom
        rdo = (Qv - re * Pv) / denom

        z = self.jfunc2(ks, ko, LAI)
        g1 = (z - J1ks * too) / (ko + m)
        g2 = (z - J1ko * tss) / (ks + m)

        Tv1 = (vf * rinf + vb) * g1
        Tv2 = (vf + vb * rinf) * g2
        T1 = Tv1 * (sf + sb * rinf)
        T2 = Tv2 * (sf * rinf + sb)
        T3 = (rdo * Qs + tdo * Ps) * rinf

        # Multiple Scattering contribution to BRDF of canopy
        rsod = (T1 + T2 - T3) / (1.0 - rinf2)

        # Hotspot-effect
        alf = torch.where(hspot > 0, ((dso / hspot) * 2.0) /
                          (ks + ko).flatten(), torch.tensor(200.0).to(device)).unsqueeze(1)
        alf[alf > 200] = 200

        fhot = LAI * torch.sqrt(ko * ks)
        fint = (1 - torch.exp(-alf)) * 0.05
        fint = fint.to(device)
        
        i19 = torch.arange(19).to(device)
        x2 = -torch.log(1.0 - (i19 + 1) * fint) / alf
        x2[:, 18] = torch.ones(x2.shape[0])  # last element in x2 is 1.0
        y2 = -(ko + ks) * LAI * x2 + fhot * (1.0 - torch.exp(-alf * x2)) / alf
        f2 = torch.exp(y2)

        # Shifts array by one and fills with constant = 0
        x1 = torch.cat((torch.zeros(x2.shape[0], 1).to(device), x2), dim=1)[:, :-1]
        y1 = torch.cat((torch.zeros(y2.shape[0], 1).to(device), y2), dim=1)[:, :-1]
        f1 = torch.cat((torch.ones(f2.shape[0], 1).to(device), f2), dim=1)[
            :, :-1]  # -"- with constant = 1
        sumint = torch.sum((f2 - f1) * (x2 - x1) / (y2 - y1), dim=1)

        tsstoo = torch.where(alf == 0, tss, f2[:, -1].unsqueeze(1))
        sumint = torch.where(alf == 0, (1.0 - tss) /
                             (ks * LAI), sumint.unsqueeze(1))

        # Bidirectional reflectance
        rsos = w * LAI * sumint  # Single scattering contribution
        dn = 1.0 - soil * rdd    # Soil interaction
        tdd_dn = tdd / dn
        # hemispherical-directional reflectance factor in viewing direction
        rdot = rdo + soil * (tdo + too) * tdd_dn
        rsodt = rsod + ((tss + tsd) * tdo +
                        (tsd + tss * soil * rdd) * too) * soil / dn
        rsost = rsos + tsstoo * soil
        rsot = rsost + rsodt  # rsot: bi-directional reflectance factor

        # Direct/diffuse light
        sin_90tts = torch.sin(math.pi / 2 - self.tts)

        if not skyl:
            skyl = 0.847 - 1.61 * sin_90tts + 1.04 * sin_90tts ** 2
        if inform_trans:
            PARdiro = (1.0 - skyl).unsqueeze(1)
            PARdifo = skyl.unsqueeze(1)
        else:
            PARdiro = torch.outer((1.0 - skyl), Es)
            PARdifo = torch.outer(skyl, Ed)

        resv = (rdot * PARdifo + rsot * PARdiro) / (PARdiro + PARdifo)

        return resv

    # Calculates the Leaf Angle Distribution Function Value (freq)
    def lidf_calc(self, LIDF, TypeLIDF):

        if TypeLIDF[0] == 1:  # Beta-Distribution, all LUT-members need to have same TypeLIDF!
            # look up frequencies for beta-distribution
            freq = beta_dict[LIDF.long(), :]
        else:  # Ellipsoidal distribution
            freq = self.campbell(LIDF)

        return freq

    # Calculates the Leaf Angle Distribution Function value (freq) Ellipsoidal distribution function from ALIA
    def campbell(self, ALIA):

        n = 13
        excent = torch.exp(-1.6184e-5 * ALIA.pow(3) + 2.1145e-3 *
                           ALIA.pow(2) - 1.2390e-1 * ALIA + 3.2491)
        freq = torch.zeros(size=(ALIA.shape[0], n)).to(device)

        x1 = excent.unsqueeze(1) / \
            torch.sqrt(1.0 + (excent.pow(2)).unsqueeze(1) * tan_tl1)
        # x1 = x1.to(device)
        
        x12 = x1.pow(2)
        x2 = excent.unsqueeze(1) / \
            torch.sqrt(1.0 + (excent.pow(2)).unsqueeze(1) * tan_tl2)
        # x2 = x2.to(device)
        
        x22 = x2.pow(2)
        alpha = excent / torch.sqrt(torch.abs(1 - excent.pow(2)))
        alpha2 = (alpha.pow(2)).unsqueeze(1).to(device)

        alpx1 = torch.sqrt(alpha2 + x12)#.to(device)
        alpx2 = torch.sqrt(alpha2 + x22)#.to(device)
        dump = x1 * alpx1 + alpha2 * torch.log(x1 + alpx1)

        almx1 = torch.sqrt(alpha2 - x12)#.to(device)
        almx2 = torch.sqrt(alpha2 - x22)#.to(device)
        
        dumm = x1 * almx1 + alpha2 * torch.asin(x1 / alpha.unsqueeze(1))
        # dumm = dumm.to(device)

        freq[excent > 1.0, :] = torch.abs(
            dump - (x2 * alpx2 + alpha2 * torch.log(x2 + alpx2)))[excent > 1.0, :].to(device)
        freq[excent < 1.0, :] = torch.abs(dumm - (x2 * almx2 + alpha2 *
                                                  torch.asin(x2 / alpha.unsqueeze(1))))[excent < 1.0, :].to(device)
        freq[excent == 1.0, :] = torch.abs(cos_tl1 - cos_tl2)

        return freq / freq.sum(dim=1).unsqueeze(1)  # Normalize

    def volscatt(self, ttl):

        costtl = torch.cos(ttl).to(device)
        sinttl = torch.sin(ttl).to(device)
        cs = torch.outer(self.costts, costtl)
        # cs = cs.to(device)
        co = torch.outer(self.costto, costtl)
        # co = co.to(device)
        ss = torch.outer(self.sintts, sinttl)
        # ss = ss.to(device)
        so = torch.outer(self.sintto, sinttl)
        # so = so.to(device)

        cosbts = torch.where(torch.abs(ss) > 1e-6, (-cs/ss), torch.tensor(5.0).to(device))
        cosbto = torch.where(torch.abs(so) > 1e-6, (-co/so), torch.tensor(5.0).to(device))
        bts = torch.where(torch.abs(cosbts) < 1,
                          torch.acos(cosbts), torch.tensor(math.pi).to(device))
        ds = torch.where(torch.abs(cosbts) < 1, ss, cs)

        chi_s = 2.0 / \
            math.pi*((bts - math.pi*0.5)
                     * cs + torch.sin(bts)*ss)

        bto = torch.where(torch.abs(cosbto) < 1, torch.acos(cosbto),
                          torch.where(self.tto.unsqueeze(1) < math.pi * 0.5, torch.tensor(math.pi).to(device), torch.tensor(0.0).to(device)))
        doo = torch.where(torch.abs(cosbto) < 1, so, torch.where(
            self.tto.unsqueeze(1) < math.pi * 0.5, co, -co))

        chi_o = 2.0 / \
            math.pi * ((bto - math.pi
                        * 0.5)*co + torch.sin(bto) * so)

        btran1 = torch.abs(bts - bto)
        btran2 = math.pi - torch.abs(bts +
                                     bto - math.pi)

        bt1 = torch.where(self.psi.unsqueeze(1) < btran1,
                          self.psi.unsqueeze(1), btran1)
        bt2 = torch.where(self.psi.unsqueeze(1) < btran1, btran1, torch.where(self.psi.unsqueeze(1)
                                                                              <= btran2, self.psi.unsqueeze(1), btran2))
        bt3 = torch.where(self.psi.unsqueeze(1) < btran1, btran2, torch.where(self.psi.unsqueeze(1)
                                                                              <= btran2, btran2, self.psi.unsqueeze(1)))

        t1 = 2 * cs * co + ss * so * self.cospsi.unsqueeze(1)
        t2 = torch.where(bt2 > 0, torch.sin(bt2) * (2 * ds * doo +
                                                    ss * so * torch.cos(bt1) * torch.cos(bt3)), torch.tensor(0.0).to(device))

        denom = 2.0 * math.pi ** 2
        frho = ((math.pi - bt2) * t1 + t2) / denom
        ftau = (-bt2 * t1 + t2) / denom

        frho[frho < 0] = 0.0
        ftau[ftau < 0] = 0.0

        return chi_s, chi_o, frho, ftau

    def jfunc1(self, k_para, l_para, t):  # J1 function with avoidance of singularity problem
        kl = k_para - l_para
        Del = kl * t
        minlt = torch.exp(-l_para * t)
        minkt = torch.exp(-k_para * t)
        Jout = torch.where(torch.abs(Del) > 1e-3,
                           (minlt - minkt) / kl,
                           0.5 * t * (minkt + minlt) * (1.0 - Del * Del / 12))
        return Jout, minkt

    def jfunc2(self, k_para, l_para, t):
        kt = k_para + l_para
        Jout = (1.0 - torch.exp(-kt * t)) / kt
        return Jout
