import torch


def gradients(input, output):
    return torch.autograd.grad(outputs=output, inputs=input, grad_outputs = torch.ones_like(output),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]




def indicator_grad(net_plus, net_minus, xr, xf, xb, pde):

    xr.requires_grad_()
    xf.requires_grad_()
    xb.requires_grad_()

    output_r = torch.where(pde.phi(xr), net_plus(xr), net_minus(xr))
    grad_r = gradients(xr, output_r)
    grad_r_magnitude = torch.norm(grad_r, dim=1)

    # #xb-grad
    output_b = net_plus(xb)
    grad_b = gradients(xb, output_b)
    grad_b_magnitude = torch.norm(grad_b, dim=1)

    #xf-grad
    grad_f_plus = gradients(xf, net_plus(xf))
    plus_magnitude = torch.norm(grad_f_plus, dim=1)
    grad_f_minus = gradients(xf, net_minus(xf))
    minus_magnitude = torch.norm(grad_f_minus, dim=1)
    max_magnitude, _ = torch.max(torch.stack([plus_magnitude, minus_magnitude], dim=1), dim=1)
    grad_f_magnitude = max_magnitude
    # grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude), dim=0)
    grad_all_magnitude = torch.cat((grad_r_magnitude, grad_f_magnitude, grad_b_magnitude), dim=0)

    return grad_all_magnitude

