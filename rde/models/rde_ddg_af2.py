import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.encoders.single import PerResidueEncoder
from rde.modules.encoders.pair import ResiduePairEncoder
from rde.modules.encoders.attn import GAEncoder
from rde.utils.protein.constants import BBHeavyAtom
from .rde import CircularSplineRotamerDensityEstimator


class DDG_RDE_Network(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Pretrain
        ckpt = torch.load(cfg.rde_checkpoint, map_location='cpu')
        self.rde = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
        self.rde.load_state_dict(ckpt['model'])
        for p in self.rde.parameters():
            p.requires_grad_(False)
        dim = ckpt['config'].model.encoder.node_feat_dim

        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=cfg.encoder.node_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB,
        )
        self.single_fusion = nn.Sequential(
            nn.Linear(2*dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.mut_bias = nn.Embedding(
            num_embeddings = 2,
            embedding_dim = dim,
            padding_idx = 0,
        )

        self.pair_encoder = ResiduePairEncoder(
            feat_dim=cfg.encoder.pair_feat_dim,
            max_num_atoms=5,    # N, CA, C, O, CB,
        )
        self.attn_encoder = GAEncoder(**cfg.encoder)

        # Pred
        self.ddg_readout = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.iptm_readout = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
        # self.plddt_readout = nn.Sequential(
        #     nn.Linear(dim, dim), nn.ReLU(),
        #     nn.Linear(dim, dim), nn.ReLU(),
        #     nn.Linear(dim, 1)
        # )
        # self.pae_readout = nn.Sequential(
        #     nn.Linear(dim, dim), nn.ReLU(),
        #     nn.Linear(dim, dim), nn.ReLU(),
        #     nn.Linear(dim, 1)
        # )

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        x_single = self.single_encoder(
            aa = batch['aa'],
            phi = batch['phi'], phi_mask = batch['phi_mask'],
            psi = batch['psi'], psi_mask = batch['psi_mask'],
            chi = chi, chi_mask = batch['chi_mask'],
            mask_residue = mask_residue,
        )
        x_pret = self._encode_rde(batch)
        x = self.single_fusion(torch.cat([x_single, x_pret], dim=-1))
        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b

        z = self.pair_encoder(
            aa = batch['aa'], 
            res_nb = batch['res_nb'], chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_atoms'], mask_atoms = batch['mask_atoms'],
        )

        x = self.attn_encoder(
            pos_atoms = batch['pos_atoms'], 
            res_feat = x, pair_feat = z, 
            mask = mask_residue
        )

        return x

    def forward(self, batch):
        
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt) # We could save some compute by only calculating this when the ddg mask is valid but let's not worry about it yet
        H_wt = h_wt.max(dim=1)[0]

        h_mt = self.encode(batch_mt)
        H_mt = h_mt.max(dim=1)[0]
        
        device = h_mt.device
        N = h_mt.shape[0]

        iptm_valid_mask = ~batch["af2_confidence_score"].isnan()
        ddg_valid_mask = ~batch["ddG"].isnan()
        # assert (ddg_valid_mask == ~iptm_valid_mask).all(), "ddg and iptm are meant to be exclusive for now"

        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)

        if ddg_valid_mask.any():
            gt_ddg = batch['ddG'][ddg_valid_mask]
            loss_ddg = (F.mse_loss(ddg_pred[ddg_valid_mask], gt_ddg, reduction="sum") + F.mse_loss(ddg_pred_inv[ddg_valid_mask], -gt_ddg, reduction="sum")) / (2 * N)
        else:
            loss_ddg = torch.zeros(1, device=device)

        iptm_logits = self.iptm_readout(H_mt).squeeze(-1)
        iptm_pred = torch.sigmoid(iptm_logits)
        if iptm_valid_mask.any():
            gt_af2_confidence_score = batch["af2_confidence_score"][iptm_valid_mask]
            loss_iptm = F.binary_cross_entropy_with_logits(iptm_logits[iptm_valid_mask], gt_af2_confidence_score, reduction="sum") / N
        else:
            loss_iptm = torch.zeros(1, device=device)

        loss_dict = {
            'iptm': loss_iptm,
            'ddg': loss_ddg,
        }
        out_dict = {
            'ddg_pred': ddg_pred,
            'ddg_true': batch['ddG'],
            'iptm_pred': iptm_pred, 
            'iptm_true': batch["af2_confidence_score"],
        }
        return loss_dict, out_dict