from typing import List, Optional

import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from diffusers.utils import logging, randn_tensor
from diffusers import DiffusionPipeline
from diffusers.pipelines.semantic_stable_diffusion import SemanticStableDiffusionPipelineOutput

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 读取一张图片并且进行预处理 512 * 512
def load_512(image_path, sizes=(512, 768), left=0, right=0, top=0, bottom=0, device=None, dtype=None):
    def pre_process(im, sizes, left=0, right=0, top=0, bottom=0):
        if type(im) is str:
            image = np.array(Image.open(im).convert('RGB'))[:, :, :3]
        elif isinstance(im, Image.Image):
            image = np.array((im).convert('RGB'))[:, :, :3]
        else:
            image = im

        h, w, c = image.shape
        left = min(left, w - 1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h - bottom, left:w - right]

        ar = max(*image.shape[:2]) / min(*image.shape[:2])

        if ar > 1.25:
            h_max = image.shape[0] > image.shape[1]
            if h_max:
                resized = Image.fromarray(image).resize((sizes[0], sizes[1]))
            else:
                resized = Image.fromarray(image).resize((sizes[1], sizes[0]))
            image = np.array(resized)

        else:
            image = np.array(Image.fromarray(image).resize((sizes[0], sizes[0])))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image

    tmps = []
    if isinstance(image_path, list):
        for item in image_path:
            tmps.append(pre_process(item, sizes, left, right, top, bottom))
    else:
        tmps.append(pre_process(image_path, sizes, left, right, top, bottom))
    image = torch.stack(tmps) / 127.5 - 1

    image = image.to(device=device, dtype=dtype)
    return image


def compute_noise_sde_dpm_pp_2nd(scheduler, prev_latents, latents, timestep, noise_pred, eta):
    def first_order_update(model_output, timestep, prev_timestep, sample):
        lambda_t, lambda_s = scheduler.lambda_t[prev_timestep], scheduler.lambda_t[timestep]
        alpha_t, alpha_s = scheduler.alpha_t[prev_timestep], scheduler.alpha_t[timestep]
        sigma_t, sigma_s = scheduler.sigma_t[prev_timestep], scheduler.sigma_t[timestep]
        h = lambda_t - lambda_s

        mu_xt = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
        )
        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))

        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample

    def second_order_update(model_output_list, timestep_list, prev_timestep, sample):
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        m0, m1 = model_output_list[-1], model_output_list[-2]
        lambda_t, lambda_s0, lambda_s1 = scheduler.lambda_t[t], scheduler.lambda_t[s0], scheduler.lambda_t[s1]
        alpha_t, alpha_s0 = scheduler.alpha_t[t], scheduler.alpha_t[s0]
        sigma_t, sigma_s0 = scheduler.sigma_t[t], scheduler.sigma_t[s0]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        mu_xt = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
        )
        sigma = sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h))

        noise = (prev_latents - mu_xt) / sigma

        prev_sample = mu_xt + sigma * noise

        return noise, prev_sample

    step_index = (scheduler.timesteps == timestep).nonzero()
    if len(step_index) == 0:
        step_index = len(scheduler.timesteps) - 1
    else:
        step_index = step_index.item()

    prev_timestep = 0 if step_index == len(scheduler.timesteps) - 1 else scheduler.timesteps[step_index + 1]

    model_output = scheduler.convert_model_output(noise_pred, timestep, latents)

    for i in range(scheduler.config.solver_order - 1):
        scheduler.model_outputs[i] = scheduler.model_outputs[i + 1]
    scheduler.model_outputs[-1] = model_output

    if scheduler.lower_order_nums < 1:
        noise, prev_sample = first_order_update(model_output, timestep, prev_timestep, latents)
    else:
        timestep_list = [scheduler.timesteps[step_index - 1], timestep]
        noise, prev_sample = second_order_update(scheduler.model_outputs, timestep_list, prev_timestep, latents)

    if scheduler.lower_order_nums < scheduler.config.solver_order:
        scheduler.lower_order_nums += 1

    return noise, prev_sample


class MyStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: DPMSolverMultistepSchedulerInject,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            requires_safety_checker: bool = True,
    ):
        super().__init__()
        if not isinstance(scheduler, DPMSolverMultistepSchedulerInject):
            scheduler = DPMSolverMultistepSchedulerInject.from_config(scheduler.config,
                                                                      algorithm_type="sde-dpmsolver++", solver_order=2)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, latents):
        #shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        #if latents.shape != shape:
        #    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def progress_bar(self, iterable=None, total=None, verbose=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )
        if not verbose:
            return iterable
        elif iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
            self,
            prompt,
            init_latents,
            zs,
            timesteps,
            guidance_scale=4.5,
            verbose=True,
            output_type: Optional[str] = "pil",
            return_dict: bool = False,
    ):
        num_images_per_prompt = 1
        latents = init_latents
        self.scheduler.model_outputs = [
                                           None,
                                       ] * self.scheduler.config.solver_order
        self.scheduler.lower_order_nums = 0

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=None)
        prompt_embeds = prompt_embeds[0]
        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        uncond_tokens: List[str]
        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=None,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}
        timesteps = timesteps[-zs.shape[0]:]
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            None,
            None,
            prompt_embeds.dtype,
            self.device,
            latents,
        )

        for i, t in enumerate(self.progress_bar(timesteps, verbose=verbose)):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            idx = t_to_idx[int(t)]
            latents = self.scheduler.step(noise_pred, t, latents, variance_noise=zs[idx]).prev_sample
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, self.device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return SemanticStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class Inversion:
    def __init__(self, model, scheduler):
        self.scheduler = scheduler
        self.model = model

    @torch.no_grad()
    def encode_image(self, image_path, dtype=None):
        image = load_512(image_path,
                         sizes=(int(self.model.unet.sample_size * self.model.vae_scale_factor),
                                int(self.model.unet.sample_size * self.model.vae_scale_factor * 1.5)),
                         device=self.model.device,
                         dtype=dtype)
        x0 = self.model.vae.encode(image).latent_dist.mode()
        x0 = self.model.vae.config.scaling_factor * x0
        return x0

    def encode_text(self, prompts):
        text_inputs = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.model.tokenizer.model_max_length:
            removed_text = self.model.tokenizer.batch_decode(text_input_ids[:, self.model.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.model.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.model.tokenizer.model_max_length]
        text_embeddings = self.model.text_encoder(text_input_ids.to(device))[0]

        return text_embeddings

    def progress_bar(self, iterable=None, total=None, verbose=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )
        if not verbose:
            return iterable
        elif iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def invert(self,
               image_path: str,
               source_prompt: str = "",
               source_guidance_scale=4.5,
               num_inversion_steps: int = 10,
               skip: float = 0.1,
               eta: float = 1.0,
               generator: Optional[torch.Generator] = None,
               verbose=True,
               ):
        self.eta = eta
        assert (self.eta > 0)

        train_steps = self.scheduler.config.num_train_timesteps
        timesteps = torch.from_numpy(
            np.linspace(train_steps - skip * train_steps - 1, 1, num_inversion_steps).astype(np.int64)).to(device)

        self.num_inversion_steps = timesteps.shape[0]
        self.scheduler.num_inference_steps = timesteps.shape[0]
        self.scheduler.timesteps = timesteps
        self.timesteps = timesteps

        # 1. get embeddings

        uncond_embedding = self.encode_text("")

        # 2. encode image
        x0 = self.encode_image(image_path, dtype=uncond_embedding.dtype)
        self.batch_size = x0.shape[0]

        if not source_prompt == "":
            text_embeddings = self.encode_text(source_prompt).repeat((self.batch_size, 1, 1))
        uncond_embedding = uncond_embedding.repeat((self.batch_size, 1, 1))
        # autoencoder reconstruction
        image_rec = self.model.vae.decode(x0 / self.model.vae.config.scaling_factor, return_dict=False)[0]
        image_rec = self.model.image_processor.postprocess(image_rec, output_type="pil")

        # 3. find zs and xts
        variance_noise_shape = (
            self.num_inversion_steps,
            *x0.shape)

        # intermediate latents
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
        xts = torch.zeros(size=variance_noise_shape, device=device, dtype=uncond_embedding.dtype)

        for t in reversed(timesteps):
            idx = self.num_inversion_steps - t_to_idx[int(t)] - 1
            noise = randn_tensor(shape=x0.shape, generator=generator, device=device, dtype=x0.dtype)
            xts[idx] = self.scheduler.add_noise(x0, noise, t)
        xts = torch.cat([x0.unsqueeze(0), xts], dim=0)

        self.scheduler.model_outputs = [
                                           None,
                                       ] * 2
        self.scheduler.lower_order_nums = 0
        # noise maps
        zs = torch.zeros(size=variance_noise_shape, device=device, dtype=uncond_embedding.dtype)

        for t in self.progress_bar(timesteps, verbose=verbose):

            idx = self.num_inversion_steps - t_to_idx[int(t)] - 1
            # 1. predict noise residual
            xt = xts[idx + 1]

            noise_pred = self.model.unet(xt, timestep=t, encoder_hidden_states=uncond_embedding).sample

            if not source_prompt == "":
                noise_pred_cond = self.model.unet(xt, timestep=t, encoder_hidden_states=text_embeddings).sample
                noise_pred = noise_pred + source_guidance_scale * (noise_pred_cond - noise_pred)

            xtm1 = xts[idx]
            z, xtm1_corrected = compute_noise_sde_dpm_pp_2nd(self.scheduler, xtm1, xt, t, noise_pred, eta)
            zs[idx] = z

            # correction to avoid error accumulation
            xts[idx] = xtm1_corrected

        # TODO: I don't think that the noise map for the last step should be discarded ?!
        # if not zs is None:
        #     zs[-1] = torch.zeros_like(zs[-1])
        self.init_latents = xts[-1].expand(self.batch_size, -1, -1, -1)
        zs = zs.flip(0)
        self.zs = zs

        return zs, xts


if __name__ == '__main__':
    img_path = "./images/18.png"
    scheduler = DPMSolverMultistepSchedulerInject.from_pretrained("/root/autodl-tmp/models/stable-diffusion-v1-5", subfolder="scheduler"
                                                             , algorithm_type="sde-dpmsolver++", solver_order=2)
    model = MyStableDiffusionPipeline.from_pretrained("/root/autodl-tmp/models/stable-diffusion-v1-5", scheduler=scheduler)
    model = model.to(device)
    inversion = Inversion(model, scheduler)
    prompt = "a cup of coffee with latte art in the shape of a heart on the foam, placed on a white saucer. The background has a blurred floral pattern, creating a warm and cozy atmosphere."
    zs, xts = inversion.invert(img_path, prompt)
    temps = {'zs':zs, 'xts':xts, 'init_latents':inversion.init_latents, 'timesteps':inversion.timesteps}
    torch.save(temps, "./temp/tensors.pth")
    init_latents = inversion.init_latents[0].float()
    init_latents = Image.fromarray(init_latents.permute(1, 2, 0).byte().cpu().numpy())
    init_latents.save("init_latents.png")
    image = model(prompt,
                  inversion.init_latents,
                  zs,
                  inversion.timesteps)[0]
    image = image[0]
    image.save("test.png")
