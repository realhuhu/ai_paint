import inspect
import torch
import prompt_parser
import k_diffusion.sampling
from k_diffusion.external import CompVisDenoiser


class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.step = 0

    def forward(self, x, sigma, uncond, cond, cond_scale):
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
        sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])

        if tensor.shape[1] == uncond.shape[1]:
            cond_in = torch.cat([tensor, uncond])
            x_out = self.inner_model(x_in, sigma_in, cond=cond_in)

        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size * 2
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])
                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=tensor[a:b])

            x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:],
                                                        cond=uncond)

        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        self.step += 1

        return denoised


class KDiffusionSampler:
    def __init__(self, funcname, sd_model):
        self.model_wrap = CompVisDenoiser(sd_model, quantize=False)
        self.funcname = funcname
        self.func = getattr(k_diffusion.sampling, self.funcname)
        self.extra_params = []
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.sampler_noise_index = 0
        self.stop_at = None
        self.eta = None
        self.default_eta = 1.0
        self.config = None

    def callback_state(self, d):
        pass

    def randn_like(self, x):
        noise = self.sampler_noises[
            self.sampler_noise_index] if self.sampler_noises is not None and self.sampler_noise_index < len(
            self.sampler_noises) else None

        if noise is not None and x.shape == noise.shape:
            res = noise
        else:
            res = torch.randn_like(x)

        self.sampler_noise_index += 1
        return res

    def initialize(self, eta):
        self.model_wrap_cfg.mask = None
        self.model_wrap_cfg.nmask = None
        self.model_wrap.step = 0
        self.sampler_noise_index = 0
        self.eta = eta

        extra_params_kwargs = {}

        if 'eta' in inspect.signature(self.func).parameters:
            extra_params_kwargs['eta'] = self.eta

        return extra_params_kwargs

    def img2img(self, init_latent, x, conditioning, unconditional_conditioning, steps, cfg, denoising_strength, eta):
        t_enc = int(min(denoising_strength, 0.999) * steps)
        sigmas = self.model_wrap.get_sigmas(steps)
        x = x * sigmas[steps - t_enc - 1]
        xi = init_latent + x

        extra_params_kwargs = self.initialize(eta)

        sigma_sched = sigmas[steps - t_enc - 1:]
        self.model_wrap_cfg.init_latent = x
        return self.func(
            self.model_wrap_cfg,
            xi,
            sigma_sched,
            extra_args={
                'cond': conditioning,
                'uncond': unconditional_conditioning,
                'cond_scale': cfg
            },
            disable=False,
            callback=self.callback_state,
            **extra_params_kwargs
        )

    def txt2img(self, x, conditioning, unconditional_conditioning, steps, cfg, eta):
        sigmas = self.model_wrap.get_sigmas(steps)

        x = x * sigmas[0]

        extra_params_kwargs = self.initialize(eta)
        if 'sigma_min' in inspect.signature(self.func).parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()
            if 'n' in inspect.signature(self.func).parameters:
                extra_params_kwargs['n'] = steps
        else:
            extra_params_kwargs['sigmas'] = sigmas
        return self.func(
            self.model_wrap_cfg,
            x,
            extra_args={
                'cond': conditioning,
                'uncond': unconditional_conditioning,
                'cond_scale': cfg
            },
            disable=False,
            callback=self.callback_state,
            **extra_params_kwargs
        )
