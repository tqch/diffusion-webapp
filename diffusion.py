import numpy as np
import torch


BETAS = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)


class GaussianDiffusion:
    def __init__(self, var_type):
        self.betas = betas = BETAS
        self.model_var_type = var_type
        self.timesteps = len(betas)

        alphas = 1 - betas
        self.alphas_bar = np.cumprod(alphas)
        self.alphas_bar_prev = np.concatenate([np.ones(1, dtype=np.float64), self.alphas_bar[:-1]])

        # q(x_t | x_{t-1})
        self.sqrt_alphas = np.sqrt(alphas)

        # q(x_t | x_0)
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)

        # q(x_{t-1} | x_t, x_0)
        # refer to the formula 1-3 in README.md
        self.sqrt_alphas_bar_prev = np.sqrt(self.alphas_bar_prev)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1. - self.alphas_bar)
        self.sqrt_recip_alphas_bar = np.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = np.sqrt(1. / self.alphas_bar - 1.)  # m1: minus 1
        self.posterior_var = betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_logvar_clipped = np.log(np.concatenate([
            np.array([self.posterior_var[1], ], dtype=np.float64), self.posterior_var[1:]]))
        self.posterior_mean_coef1 = betas * self.sqrt_alphas_bar_prev / (1. - self.alphas_bar)
        self.posterior_mean_coef2 = np.sqrt(alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)

        # for fixed model_var_type's
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large": (self.betas, np.log(np.concatenate([np.array([self.posterior_var[1]]), self.betas[1:]]))),
            "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
        }[self.model_var_type]

    @staticmethod
    def _extract(arr, t, ndim):
        B = len(t)
        out = torch.as_tensor(arr, dtype=torch.float32, device=t.device).gather(0, t)
        return out.reshape((B,) + (1,) * (ndim - 1))

    def q_mean_var(self, x_0, t):
        ndim = x_0.ndim
        mean = self._extract(self.sqrt_alphas_bar, t, ndim=ndim) * x_0
        var = self._extract(1. - self.alphas_bar, t, ndim=ndim)
        logvar = self._extract(self.sqrt_one_minus_alphas_bar, t, ndim=ndim)
        return mean, var, logvar

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        ndim = x_0.ndim
        coef1 = self._extract(self.sqrt_alphas_bar, t, ndim=ndim).to(x_0.device)
        coef2 = self._extract(self.sqrt_one_minus_alphas_bar, t, ndim=ndim).to(x_0.device)
        return coef1 * x_0 + coef2 * noise

    def q_posterior_mean_var(self, x_0, x_t, t):
        ndim = x_0.ndim
        posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, ndim=ndim).to(x_0.device)
        posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, ndim=ndim).to(x_0.device)
        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t
        posterior_var = self._extract(self.posterior_var, t, ndim=ndim)
        posterior_logvar = self._extract(self.posterior_logvar_clipped, t, ndim=ndim).to(x_0.device)
        return posterior_mean, posterior_var, posterior_logvar

    def p_mean_var(self, denoise_fn, x_t, t, clip_denoised, return_pred):
        ndim = x_t.ndim
        out = denoise_fn(x_t, t)

        model_var, model_logvar = self._extract(self.fixed_model_var, t, ndim=ndim),\
                                  self._extract(self.fixed_model_logvar, t, ndim=ndim)
        model_var, model_logvar = model_var.to(x_t.device), model_logvar.to(x_t.device)

        # calculate the mean estimate
        _clip = (lambda x: x.clamp(-1., 1.)) if clip_denoised else (lambda x: x)
        pred_x_0 = _clip(self._pred_x_0_from_eps(x_t=x_t, eps=out, t=t))
        model_mean, *_ = self.q_posterior_mean_var(x_0=pred_x_0, x_t=x_t, t=t)

        if return_pred:
            return model_mean, model_var, model_logvar, pred_x_0
        else:
            return model_mean, model_var, model_logvar

    def _pred_x_0_from_mean(self, x_t, mean, t):
        ndim = x_t.ndim
        coef1 = self._extract(self.posterior_mean_coef1, t, ndim=ndim).to(x_t.device)
        coef2 = self._extract(self.posterior_mean_coef2, t, ndim=ndim).to(x_t.device)
        return mean / coef1 - coef2 / coef1 * x_t

    def _pred_x_0_from_eps(self, x_t, eps, t):
        ndim = x_t.ndim
        coef1 = self._extract(self.sqrt_recip_alphas_bar, t, ndim=ndim).to(x_t.device)
        coef2 = self._extract(self.sqrt_recip_m1_alphas_bar, t, ndim=ndim).to(x_t.device)
        return coef1 * x_t - coef2 * eps

    # === sample ===

    def p_sample_step(self, denoise_fn, x_t, t, clip_denoised=True, return_pred=False):
        ndim = x_t.ndim
        model_mean, _, model_logvar, pred_x_0 = self.p_mean_var(
            denoise_fn, x_t, t, clip_denoised=clip_denoised, return_pred=True)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).reshape((-1,) + (1,) * (ndim - 1)).to(x_t)
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_logvar) * noise
        return (sample, pred_x_0) if return_pred else sample

    @torch.inference_mode()
    def p_sample(self, denoise_fn, noise):
        x_t = noise
        t = torch.empty((noise.shape[0], ), dtype=torch.int64, device=noise.device)
        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(denoise_fn, x_t, t)
        return x_t.cpu()

    @torch.inference_mode()
    def p_sample_progressive(self, denoise_fn, noise, pred_freq=50):
        x_t = noise
        L = self.timesteps // pred_freq
        preds = torch.zeros((L, ) + noise.shape, dtype=torch.float32)
        idx = L
        t = torch.empty(noise.shape[0], dtype=torch.int64, device=noise.device)
        for ti in range(self.timesteps - 1, -1, -1):
            t.fill_(ti)
            x_t, pred = self.p_sample_step(denoise_fn, x_t, t, return_pred=True)
            if (ti + 1) % pred_freq == 0:
                idx -= 1
                preds[idx] = pred.cpu()
        return x_t.cpu(), preds


def get_selection_schedule(schedule, size, timesteps):
    """
    :param schedule: selection schedule
    :param size: length of subsequence
    :param timesteps: total timesteps of pretrained ddpm model
    :return: subsequence
    """
    assert schedule in {"linear", "quadratic"}

    if schedule == "linear":
        subsequence = np.arange(0, timesteps, timesteps // size)
    else:
        subsequence = np.power(np.linspace(0, np.sqrt(timesteps * 0.8), size), 2).astype(np.int32)

    return subsequence


class GeneralizedDiffusion(GaussianDiffusion):
    def __init__(self, model_var_type, eta, subseq_size, schedule):
        super().__init__(model_var_type)
        self.eta = eta  # coefficient between [0, 1] that decides the behavior of generative process
        self.subsequence = subsequence = get_selection_schedule(
            schedule, subseq_size, self.timesteps)  # subsequence of the accelerated generation

        eta2 = eta ** 2
        assert not (eta2 != 1. and model_var_type != "fixed-small"),\
            'Cannot use DDIM (eta < 1) with var type other than "fixed-small"'

        self.alphas_bar = self.alphas_bar[subsequence]
        self.alphas_bar_prev = np.concatenate([np.ones(1, dtype=np.float64), self.alphas_bar[:-1]])
        self.alphas = self.alphas_bar / self.alphas_bar_prev
        self.betas = 1. - self.alphas
        self.sqrt_alphas_bar_prev = np.sqrt(self.alphas_bar_prev)

        # q(x_t|x_0)
        # re-parameterization: x_t(x_0, \epsilon_t)
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1. - self.alphas_bar)

        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar) * eta2
        self.posterior_logvar_clipped = np.log(np.concatenate([
            np.array([self.posterior_var[1], ], dtype=np.float64), self.posterior_var[1:]]).clip(min=1e-20))

        # coefficients to recover x_0 from x_t and \epsilon_t
        self.sqrt_recip_alphas_bar = np.sqrt(1. / self.alphas_bar)
        self.sqrt_recip_m1_alphas_bar = np.sqrt(1. / self.alphas_bar - 1.)

        # coefficients to calculate E[x_{t-1}|x_0, x_t]
        self.posterior_mean_coef2 = np.sqrt(
            1 - self.alphas_bar - eta2 * self.betas
        ) * np.sqrt(1 - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_mean_coef1 = self.sqrt_alphas_bar_prev * (1. - np.sqrt(self.alphas) * self.posterior_mean_coef2)

        # for fixed model_var_type's
        self.fixed_model_var, self.fixed_model_logvar = {
            "fixed-large": (
                self.betas, np.log(
                np.concatenate([np.array([self.posterior_var[1]]), self.betas[1:]]).clip(min=1e-20))),
            "fixed-small": (self.posterior_var, self.posterior_logvar_clipped)
        }[self.model_var_type]

        self.subsequence = torch.as_tensor(subsequence)

    @torch.inference_mode()
    def p_sample(self, denoise_fn, noise):
        x_t = noise
        S = len(self.subsequence)
        subsequence = self.subsequence.to(noise.device)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))
        t = torch.empty((noise.shape[0], ), dtype=torch.int64, device=noise.device)
        for ti in range(S - 1, -1, -1):
            t.fill_(ti)
            x_t = self.p_sample_step(_denoise_fn, x_t, t)
        return x_t

    @torch.inference_mode()
    def p_sample_progressive(self, denoise_fn, noise, pred_freq=1):
        x_t = noise
        S = len(self.subsequence)
        subsequence = self.subsequence.to(noise.device)
        idx = L = S // pred_freq
        preds = torch.zeros((L, ) + noise.shape, dtype=torch.float32)
        _denoise_fn = lambda x, t: denoise_fn(x, subsequence.gather(0, t))
        t = torch.empty(noise.shape[0], dtype=torch.int64, device=noise.device)
        for ti in range(S - 1, -1, -1):
            t.fill_(ti)
            x_t, pred = self.p_sample_step(_denoise_fn, x_t, t, return_pred=True)
            if (ti + 1) % pred_freq == 0:
                idx -= 1
                preds[idx] = pred.cpu()
        return x_t.cpu(), preds
