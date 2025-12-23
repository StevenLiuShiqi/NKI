import torch


def piecewise_safe_division(a, b):
    """
    Piecewise safe division of tensors a/b with special rules for division by zero.
    Unravels/flattens `a` and `b` if not already 1-D (handled by summary_stats).
    """
    is_zero = b == 0

    # Compute x/y where y != 0
    div = a / b

    # For x==0 & y==0 --> 0
    case1 = (a == 0) & is_zero

    # For x<0 & y==0 --> -inf
    case2 = (a < 0) & is_zero

    # For x>0 & y==0 --> inf
    case3 = (a > 0) & is_zero

    # Start with normal division
    c = div

    # Set 0 where x==0 & y==0
    c = torch.where(case1, torch.zeros_like(c), c)

    # Set -inf where x<0 & y==0
    c = torch.where(case2, torch.full_like(c, -float('inf')), c)

    # Set inf where x>0 & y==0
    c = torch.where(case3, torch.full_like(c, float('inf')), c)

    return c


def summary_stats(A, B):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        with torch.no_grad():
            # 1. Check matching shape
            if A.shape != B.shape:
                raise ValueError(f"Tensor shape mismatch: {A.shape} vs {B.shape}")

            # 2. Flatten inputs
            A = A.contiguous().view(-1)
            B = B.contiguous().view(-1)

            diff = A - B

            abs_error = torch.abs(diff)
            min_abs_error = torch.min(abs_error)
            median_abs_error = torch.median(abs_error)
            max_abs_error = torch.max(abs_error)

            denominator = torch.abs(A) + torch.abs(B)
            rel_error = piecewise_safe_division(abs_error, denominator)
            min_rel_error = torch.min(rel_error)
            median_rel_error = torch.median(rel_error)
            max_rel_error = torch.max(rel_error)
            
            print("Tensor Comparison Results:\n")
            print(f"Absolute Error (min/median/max):  {min_abs_error:.4g}/{median_abs_error:.4g}/{max_abs_error:.4g}")
            print(f"Relative Error (min/median/max):  {min_rel_error:.4g}/{median_rel_error:.4g}/{max_rel_error:.4g}")
    elif isinstance(A, tuple) and isinstance(B, tuple):
        if len(A) != len(B):
            raise ValueError(f"Tuple length mismatch: {len(A)} vs {len(B)}")

        for i, (a, b) in enumerate(zip(A, B)):
            print(f"Element {i}:")
            print()

            summary_stats(a, b)
            print()
    else:
        raise ValueError(f"Unknown and/or mismatching types: {type(A)} and {type(B)}")
        