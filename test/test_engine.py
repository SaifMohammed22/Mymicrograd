import torch
import sys
import os
from time import monotonic

# Add the parent directory to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from micrograd.engine import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    t_s = monotonic()
    y.backward()
    t_e = monotonic()
    print(f"Time taken by backward function (Micrograd): {t_e - t_s:.5f}")
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    t_s = monotonic()
    y.backward()
    t_e = monotonic()
    print(f"Time taken by backward function (Pytorch): {t_e - t_s:.5f}")
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    if ymg.data == ypt.data.item():
        print("Y grad test passed successfully!")
    else:
        print("Something went wrong, the values are different")
    # backward pass went well
    assert xmg.grad == xpt.grad.item()
    if xmg.grad == xpt.grad.item():
        print("X grad test passed successfully!")
    else:
        print("Something went wrong, the values are different")



def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

if __name__ == "__main__":
    test_sanity_check()