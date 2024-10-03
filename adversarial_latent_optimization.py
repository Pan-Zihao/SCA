import os
from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from diffusers.utils import logging
from diffusers import DiffusionPipeline

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from get_model import get_model

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

seed = 20240404

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.Generator().manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epsilon = 0.001
beta = 0
alpha = 0.04
eps = 0.1
mu = 1
norm = 2
image_size = (224, 224)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
mean = torch.Tensor(mean).cuda()
std = torch.Tensor(std).cuda()
model = input("target model name, resnet50 or mnv2")
net = get_model(model)
if device == 'cuda':
    net.to(device)
    cudnn.benchmark = True
net.eval()
net.cuda()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_dir = BASE_DIR
writer = SummaryWriter(log_dir=log_dir, filename_suffix="tensorboard")

def limitation01(y):
    idx = (y > 1)
    y[idx] = (torch.tanh(1000 * (y[idx] - 1)) + 10000) / 10001
    idx = (y < 0)
    y[idx] = (torch.tanh(1000 * (y[idx]))) / 10000
    return y


def norm_l2(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


class adversarial_latent_optimization(DiffusionPipeline):
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
        # shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        # if latents.shape != shape:
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
    def denoising(
            self,
            prompt,
            init_latents,
            zs,
            timesteps,
            guidance_scale=4.5,
            verbose=True,
    ):
        self.prompt = prompt
        self.init_latents = init_latents
        self.zs = zs
        self.timesteps = timesteps
        self.scheduler.num_inference_steps = timesteps.shape[0]
        self.scheduler.timesteps = timesteps
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        return latents

    @torch.no_grad()
    def optimization(self,
                     adv_steps=10,
                     label=None,
                     raw_img=None):

        latents = self.init_latents
        best_latents = latents
        ori_latents = latents.clone().detach()
        adv_latents = latents.clone().detach()
        success = True
        momentum = 0
        for k in tqdm(range(adv_steps)):
            latents = adv_latents
            latents = self.denoising(self.prompt, latents, self.zs, self.timesteps)
            gradient = torch.zeros_like(latents)
            '''
            for i in range(10):
                rand_result = self.denoising(self.prompt, ori_latents + epsilon * torch.randn_like(ori_latents), self.zs, self.timesteps)
                gradient_i = (rand_result - latents)/epsilon
                gradient += gradient_i
            gradient /= 10
            '''
            #print(gradient.size())
            image = None
            with torch.enable_grad():
                latents_last = latents.detach().clone()
                latents_last.requires_grad = True
                latents_t = (1 / 0.18215 * latents_last)
                image = self.vae.decode(latents_t)['sample']
                image = (image / 2 + 0.5)

                image = limitation01(image)
                image_m = F.interpolate(image, image_size)

                image_m = image_m - mean[None, :, None, None]
                image_m = image_m / std[None, :, None, None]
                outputs = net(image_m)
                _, predicted = outputs.max(1)

                if label != predicted:
                    best_latent = adv_latents
                    success = False
                loss_ce = torch.nn.CrossEntropyLoss()(outputs, torch.Tensor([label]).long().cuda())

                #loss_mse = beta * torch.norm(image_m - raw_img, p=norm).mean()
                #loss = loss_ce - loss_mse
                loss = loss_ce
                writer.add_scalar("loss/sum", loss, k)
                writer.add_scalar("loss/ce", loss_ce, k)
                #writer.add_scalars("loss/ce-mse",{'loss_ce':loss_ce,'loss_mse':loss_mse},k)
                loss.backward()

                print('*' * 50)
                #print('Loss', loss.item(), 'Loss_ce', loss_ce.item(), 'Loss_mse', loss_mse.item())
                print('Loss', loss.item(), 'Loss_ce', loss_ce.item())
                print(k, 'Predicted:', label, predicted, loss.item())
                
            #latents_last_grad = latents_last.grad * gradient
            
            l1_grad = latents_last.grad / torch.norm(latents_last.grad, p=1)
            momentum = mu * momentum + l1_grad
            adv_latents = adv_latents + torch.sign(momentum) * alpha
            noise = (adv_latents - ori_latents).clamp(-eps, eps)
            adv_latents = ori_latents + noise
            latents = adv_latents.detach()

        if success:
            best_latent = latents

        latents = best_latent
        latents = (1 / 0.18215 * latents)
        image = self.vae.decode(latents_t)['sample']
        image = (image / 2 + 0.5)
        image = limitation01(image)
        image = F.interpolate(image, image_size)
        image = image.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        writer.add_image("image_inv:{}".format(image[0].shape),image[0],0,dataformats='HWC')
        return image, best_latent, success


if __name__ == '__main__':
    loaded = torch.load("./temp/tensors.pth")
    #loaded = loaded.to(device)
    zs = loaded['zs']
    xts = loaded['xts']
    init_latents = loaded['init_latents']
    timesteps = loaded['timesteps']
    prompt = "a cup of coffee with latte art in the shape of a heart on the foam, placed on a white saucer. The background has a blurred floral pattern, creating a warm and cozy atmosphere."
    scheduler = DPMSolverMultistepSchedulerInject.from_pretrained("/root/autodl-tmp/models/stable-diffusion-v1-5", subfolder="scheduler"
                                                             , algorithm_type="sde-dpmsolver++", solver_order=2)
    pipe = adversarial_latent_optimization.from_pretrained("/root/autodl-tmp/models/stable-diffusion-v1-5", scheduler=scheduler)
    pipe = pipe.to(device)
    _ = pipe.denoising(prompt, init_latents, zs, timesteps)
    raw_img_path = "./images/18.png"
    label = 967
    pil_image = Image.open(raw_img_path).convert('RGB').resize(image_size)
    raw_img = (torch.tensor(np.array(pil_image), device=device).unsqueeze(0) / 255.).permute(0, 3, 1, 2)
    raw_img = raw_img - mean[None, :, None, None]
    raw_img = raw_img / std[None, :, None, None]
    image_inv, x_t, success = pipe.optimization(10, label, raw_img)
    
    print(success)
    writer.close()
    img = Image.fromarray(image_inv[0])
    img.save('output_image.png')

    



