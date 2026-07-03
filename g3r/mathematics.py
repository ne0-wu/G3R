import torch


def fund_sol(x, y):
    """
    Compute the fundamental solution to Laplace's equation in 3D.

    This function implements the formula:
        FundamentalSol3d[x, y] = 1/(4 Pi ||x - y||).

    Parameters:
    - x: n*3 tensor of points (x1, x2, x3)
    - y: m*3 tensor of points (y1, y2, y3)

    Returns:
    - n*m tensor of the fundamental solution values
    """
    r = torch.linalg.norm(x[:, None, :] - y[None, :, :], dim=-1)
    r = torch.maximum(r, torch.tensor(1e-10, device=r.device))  # avoid division by zero
    return 1 / (4 * torch.pi * r)


def fund_sol_grad_y(x, y):
    """
    Compute the gradient of the fundamental solution w.r.t. y.

    This function computes the gradient of the fundamental solution w.r.t. y:
        GradFundamentalSol[y] = 1/(4 Pi) * (x - y) / ||x - y||³.

    Parameters:
    - x: n*3 tensor of points (x1, x2, x3)
    - y: m*3 tensor of points (y1, y2, y3)

    Returns:
    - n*m*3 tensor of the gradient of the fundamental solution w.r.t. y
    """
    diff = x[:, None, :] - y[None, :, :]
    r = torch.linalg.norm(diff, dim=-1)
    r = torch.maximum(r, torch.tensor(1e-10, device=r.device))  # avoid division by zero

    return 1 / (4 * torch.pi) * diff / (r[..., None] ** 3)


def fund_sol_eps(x, y, epsilon: float = 1e-3):
    """
    Compute the regularized fundamental solution to Laplace's equation in 3D.

    This function implements the formula:
        RegularizedFundamentalSol3d[x, y] = Erf[||x - y||/(√2 ε)]/(4 Pi ||x - y||).

    Parameters:
    - x: n*3 tensor of points (x1, x2, x3)
    - y: m*3 tensor of points (y1, y2, y3)
    - epsilon: regularization parameter (ε in the formula)

    Returns:
    - n*m tensor of the regularized fundamental solution values
    """
    # Compute the distance between points x and y
    r = torch.linalg.norm(x[:, None, :] - y[None, :, :], dim=-1)
    r = torch.maximum(r, torch.tensor(1e-10, device=r.device))  # avoid division by zero

    # Compute Erf[r/(√2 ε)] / (4π r)
    erf_term = torch.erf(r / (torch.sqrt(torch.tensor(2.0, device=r.device)) * epsilon))

    return erf_term / (4 * torch.pi * r)


def fund_sol_eps_grad_y(x, y, epsilon: float = 1e-3):
    """
    Compute the gradient of the regularized fundamental solution in 3D w.r.t. y.

    This function computes the gradient of the regularized fundamental solution w.r.t. y:
        GradFundamentalSolEps[y] = Regularizer3d[||x-y||/ε] * PoissonKernel3d[x, y, n].

    Where Regularizer3d[r] = -E^(-r²/2) * √(2/π) * r + Erf[r/√2]

    Parameters:
    - x: n*3 tensor of points (x1, x2, x3)
    - y: m*3 tensor of points (y1, y2, y3)
    - epsilon: regularization parameter (ε in the formula)

    Returns:
    - n*m*3 tensor of the gradient values with respect to y
    """
    # Compute the difference vector (x - y)
    diff = x[:, None, :] - y[None, :, :]

    # Compute the distance
    r = torch.linalg.norm(diff, dim=-1)
    r = torch.maximum(r, torch.tensor(1e-10, device=r.device))  # avoid division by zero

    # Compute the regularizer Regularizer3d[r/ε] = -E^(-(r/ε)²/2) * √(2/π) * (r/ε) + Erf[(r/ε)/√2]
    r_scaled = r / epsilon
    sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0, device=r.device) / torch.pi)
    exp_term = -torch.exp(-(r_scaled**2) / 2) * sqrt_2_over_pi * r_scaled
    erf_term = torch.erf(r_scaled / torch.sqrt(torch.tensor(2.0, device=r.device)))
    regularizer = exp_term + erf_term

    # Compute the Poisson kernel part: 1/(4π) * (x-y) / ||x-y||³
    poisson_kernel = 1 / (4 * torch.pi) * diff / (r[..., None] ** 3)

    # Combine: Regularizer * PoissonKernel
    grad_y = regularizer[..., None] * poisson_kernel

    return grad_y


def gaussian_kernel_3d(x, epsilon: float = 1e-3):
    """
    Compute the 3D Gaussian kernel.

    This function implements the formula:
        GaussianKernel3d[x] = E^(-x·x/(2ε²)) / (√(2π) ε)³

    Parameters:
    - x: n*3 tensor of points (x1, x2, x3)
    - epsilon: regularization parameter (ε in the formula)

    Returns:
    - n tensor of Gaussian kernel values
    """
    # Compute x·x (dot product for each point)
    x_dot_x = torch.sum(x**2, dim=-1)

    # Compute the Gaussian kernel
    sqrt_2_pi = torch.sqrt(torch.tensor(2 * torch.pi, device=x.device))
    gaussian = torch.exp(-x_dot_x / (2 * epsilon**2)) / ((sqrt_2_pi * epsilon) ** 3)

    return gaussian
