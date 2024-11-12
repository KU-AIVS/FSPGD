import torch
import math


class Attack:
    def __init__(self):
        pass

    @staticmethod
    def step_inf(
            perturbed_image,
            epsilon,
            data_grad,
            orig_image,
            alpha,
            clamp_min=0,
            clamp_max=1,
            grad_scale=None
    ):
        sign_data_grad = alpha * data_grad.sign()
        if grad_scale is not None:
            sign_data_grad *= grad_scale
        perturbed_image = perturbed_image.detach() + sign_data_grad
        delta = perturbed_image - orig_image
        delta = torch.permute(delta, (0, 2, 3, 1))
        delta = torch.clamp(delta, min=-torch.tensor(epsilon).to(orig_image.device),
                            max=torch.tensor(epsilon).to(orig_image.device))
        delta = torch.permute(delta, (0, 3, 1, 2))

        perturbed_image = orig_image + delta
        perturbed_image = torch.permute(perturbed_image, (0, 2, 3, 1))
        perturbed_image = torch.clamp(perturbed_image, torch.tensor(clamp_min).to(orig_image.device),
                                      torch.tensor(clamp_max).to(orig_image.device)).detach()
        perturbed_image = torch.permute(perturbed_image, (0, 3, 1, 2))

        return perturbed_image


    @staticmethod
    def step_l2(
            perturbed_image,
            epsilon,
            data_grad,
            orig_image,
            alpha,
            clamp_min = 0,
            clamp_max = 1,
            grad_scale = None
        ):

        data_grad = Attack.lp_normalize(
            data_grad,
            p = 2,
            epsilon = 1.0,
            decrease_only = False
        )
        if grad_scale is not None:
            data_grad *= grad_scale
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = perturbed_image.detach() + alpha*data_grad
        # clip to l2 ball
        delta = Attack.lp_normalize(
            noise = perturbed_image - orig_image,
            p = 2,
            epsilon = epsilon,
            decrease_only = True
        )
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()
        return perturbed_image
    @staticmethod
    def lp_normalize(
            noise,
            p,
            epsilon = None,
            decrease_only = False
        ):
        if epsilon is None:
            epsilon = torch.tensor(1.0)
        denom = torch.norm(noise, p=p, dim=(-1, -2, -3))
        denom = torch.maximum(denom, torch.tensor(1E-12)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if decrease_only:
            denom = torch.maximum(denom/epsilon, torch.tensor(1))
        else:
            denom = denom / epsilon
        return noise / denom
    @staticmethod
    def init_linf(
            images,
            epsilon,
            clamp_min=0,
            clamp_max=1,
        ):
        noise = torch.zeros_like(images)

        for i, eps in enumerate(epsilon):
            noise[:, i] = noise[:, i].uniform_(-eps, eps)
        images = images + noise
        images = torch.permute(images, (0, 2, 3, 1))
        images = torch.clamp(images, torch.tensor(clamp_min).to(images.device), torch.tensor(clamp_max).to(images.device))
        images = torch.permute(images, (0, 3, 1, 2))
        return images


    @staticmethod
    def init_l2(
            images,
            epsilon,
            clamp_min = 0,
            clamp_max = 1,
        ):
        noise = torch.empty_like(images).uniform_(-1, 1).to(images.device) * epsilon
        noise = Attack.lp_normalize(
            noise = noise,
            p = 2,
            epsilon = epsilon,
            decrease_only = False
        )
        images = images + noise
        images = images.clamp(clamp_min, clamp_max)
        return images


    @staticmethod
    def fspgd(
            mid_original,
            mid_adv,
            cosine,
            iteration,
            iterations,
        ):
        lambda_t = iteration / iterations
        n, c, h, w = mid_original.size()
        mid_original = torch.nn.functional.normalize(mid_original, dim=1)  # original feature map normalize
        f_orig = torch.reshape(mid_original, (n, c, h * w))  # C X (H*W) reshape
        del mid_original
        W = torch.bmm(torch.transpose(f_orig, 1, 2), f_orig)
        W = ((W * (torch.ones((n, h * w, h * w)) - torch.eye(h * w)).cuda()) > torch.cos(
            torch.tensor(math.pi / cosine))).float().cuda()

        f_adv = torch.reshape(mid_adv, (n, c, h * w))
        f_diag = torch.nn.CosineSimilarity(dim=1, eps=1e-8)(f_orig, f_adv)
        f_combi = (torch.sum(W * torch.bmm(torch.transpose(f_adv, 1, 2), f_adv)) / torch.count_nonzero(W)) / 2
        loss = - lambda_t * f_diag.mean() - (1 - lambda_t) * f_combi.mean()

        return loss



