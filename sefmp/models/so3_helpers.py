import torch


def exp(A):
    return torch.linalg.matrix_exp(A)


def expmap(R0, tangent):
    skew_sym = torch.einsum("...ij,...ik->...jk", R0, tangent)
    return torch.einsum("...ij,...jk->...ik", R0, exp(skew_sym))


# Parallel Transport a matrix at v at point R to the Tangent Space at identity
def pt_to_identity(R, v):
    return torch.transpose(R, dim0=-2, dim1=-1) @ v


def norm_SO3(R, T_R):
    # calulate the norm squared of matrix T_R in the tangent space of R
    r = pt_to_identity(R, T_R)  # matrix r is in so(3)
    norm = -torch.diagonal(r @ r, dim1=-2, dim2=-1).sum(dim=-1) / 2  # -trace(rTr)/2
    return norm


def tangent_space_proj(R, M):
    """
    Project the given 3x3 matrix M onto the tangent space of SO(3) at point R in PyTorch.

    Args:
    - M (torch.Tensor): 3x3 matrix from R^9
    - R (torch.Tensor): 3x3 matrix from SO(3) representing the point of tangency

    Returns:
    - T (torch.Tensor): projected 3x3 matrix in the tangent space of SO(3) at R
    """
    # Compute the skew-symmetric part of M
    skew_symmetric_part = 0.5 * (M - M.permute(0, 2, 1))

    # Project onto the tangent space at R
    T = R @ skew_symmetric_part

    return T


def loss_so3(v, u, x):
    res = v - u
    norm = norm_SO3(x, res)  # norm-squared on SO(3)
    loss = torch.mean(norm, dim=-1)
    return loss
