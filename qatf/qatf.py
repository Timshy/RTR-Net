import numpy as np
from .topology_utils import binarize, dilate, skeleton_connectivity


def compute_quality_cues(response_map, rtr_prob, response_threshold=0.50, rtr_threshold=0.50, near_addition_radius=20, eps=1e-6):
    """Compute QATF cues.

    B_t: binarized original response.
    C_t: binarized RTR-Net prediction.

    p_near follows the paper definition: newly added rectified regions
    (C_t \ B_t) should be near the original response B_t.
    """
    b = binarize(response_map, response_threshold)
    c = binarize(rtr_prob, rtr_threshold)

    deleted = (b == 1) & (c == 0)
    added = (c == 1) & (b == 0)

    p_del = float(deleted.sum()) / (float(b.sum()) + eps)
    delta_c = 100.0 * (skeleton_connectivity(c, eps) - skeleton_connectivity(b, eps))

    support = dilate(b, near_addition_radius).astype(bool)
    p_near = float((added & support).sum()) / (float(added.sum()) + eps)

    return {'B': b, 'C': c, 'p_del': p_del, 'delta_c': delta_c, 'p_near': p_near}


def select_route(p_del, delta_c, p_near, eta_c=11.0, eta_n=0.62, eta_d=0.41, gamma_c=40.0, gamma_d=0.60):
    if (delta_c >= eta_c) and (p_near >= eta_n) and (p_del <= eta_d):
        base = 'CTR'
    else:
        base = 'CTI'
    if (base == 'CTI') and (delta_c <= gamma_c) and (p_del >= gamma_d):
        return 'PCTR'
    return base


def apply_route(b, c, route, pctr_preservation_radius=30):
    if route == 'CTR':
        return c.astype(np.uint8)
    if route == 'CTI':
        return np.logical_or(b, c).astype(np.uint8)
    if route == 'PCTR':
        near_c = dilate(c, pctr_preservation_radius).astype(bool)
        preserved = ((b == 1) & (c == 0) & near_c)
        return np.logical_or(c.astype(bool), preserved).astype(np.uint8)
    raise ValueError(f'Unknown QATF route: {route}')


def qatf_fusion(response_map, rtr_prob, cfg):
    cues = compute_quality_cues(
        response_map,
        rtr_prob,
        response_threshold=cfg.get('response_threshold', 0.50),
        rtr_threshold=cfg.get('rtr_threshold', 0.50),
        near_addition_radius=cfg.get('near_addition_radius', 20),
    )
    route = select_route(
        cues['p_del'], cues['delta_c'], cues['p_near'],
        eta_c=cfg.get('eta_c', 11.0),
        eta_n=cfg.get('eta_n', 0.62),
        eta_d=cfg.get('eta_d', 0.41),
        gamma_c=cfg.get('gamma_c', 40.0),
        gamma_d=cfg.get('gamma_d', 0.60),
    )
    final = apply_route(cues['B'], cues['C'], route, cfg.get('pctr_preservation_radius', 30))
    cues['route'] = route
    return final, cues
