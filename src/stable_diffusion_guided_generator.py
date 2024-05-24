from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline

from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *
from omegaconf import OmegaConf

@dataclass
class Config():
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    enable_memory_efficient_attention: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_channels_last_format: bool = False
    grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    half_precision_weights: bool = True

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    max_step_percent_annealed: float = 0.5
    anneal_start_step=None

    use_sjc: bool = False
    var_red: bool = True
    weighting_strategy: str = "sds"

    token_merging: bool = False
    token_merging_params: Optional[dict] =field(default_factory=dict)

    view_dependent_prompting: bool = False

    """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
    max_items_eval: int = 4
    
    cache_dir: Optional[str] = None

    ip2p_name_or_path: str = "timbrooks/instruct-pix2pix"

    guidance_scale: float = 7.5
    condition_scale: float = 1.5
    half_precision_weights: bool = True

    fixed_size: int = -1
    diffusion_steps: int = 20

    use_sds: bool = False
        
class StableDiffusionGuidedGenerator():
    def __init__(self, cfg, device):
        self.device = device 
        self.cfg = OmegaConf.structured(Config(**cfg)) #merge the configuation
        self.configure()

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.cfg.ip2p_name_or_path, **pipe_kwargs
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        # for p in self.vae.parameters():
        #     p.requires_grad_(False)
        # for p in self.unet.parameters():
        #     p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.5)  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        # self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        self.scheduler.timesteps  = self.scheduler.timesteps[self.cfg.diffusion_steps//2:]
        # add noise
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, t//2)  # type: ignore
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual with unet, NO grad!
            # pred noise
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = torch.cat(
                [latent_model_input, image_cond_latents], dim=1
            )

            noise_pred = self.forward_unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                3
            )
            noise_pred = (
                noise_pred_uncond
                + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents


    @torch.cuda.amp.autocast(enabled=False)
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs] 

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
    
    
    def __call__(
        self,
        input_image: Float[Tensor, "B 3 H W"],
        # latents : Float[Tensor, "B 4 DH DW"],
        # cond_latents: Float[Tensor, "B H RW RC"],
        text_embeddings:  Float[Tensor, "B 77 768"],
        # prompt_utils: PromptProcessorOutput,
        **kwargs,
    ):
        latents = self.encode_images(input_image) # latent_codes: "B 4 64 64"
        batch_size = input_image.shape[0]
        text_embeddings = torch.cat(
            [text_embeddings, text_embeddings[-1:]], dim=0
        )  # [positive, negative, negative]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        cond_rgb_BCHW_HW8 = input_image.clone()
        with torch.no_grad():
            cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)
        edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t)
        edit_images = self.decode_latents(edit_latents)
        edit_images = F.interpolate(edit_images, (512, 512), mode="bilinear")

        return {"edit_images": edit_images}
        
if __name__ == "__main__":
 # "pretrained_model_name": "pretrained_model/miniSD.ckpt",
    cfg = {"pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "diffusion_steps": 4,
            "guidance_scale": 7.5,
            "min_step_percent": 0.02,
            "max_step_percent": 0.30}
    device = torch.device("cuda:1")
    stable_diffusion_generator = StableDiffusionGuidedGenerator(cfg, device)
    # Read a single image
    from PIL import Image
    import torchvision.transforms as transforms
    from stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
    image_path = "/home/yjli/AIGC/Adversarial_camou/results/experiment/three dogs, three cats and three trees/texture-diffusion-three dogs, three cats and three trees-best-iter88.png"
    # image_path = "/home/yjli/AIGC/Adversarial_camou/results/clothes/PNG/Front.png"#
    image = Image.open(image_path)
    # Define the desired image size for Stable Diffusion
    desired_size = (512, 512)
    H = desired_size[0]
    W = desired_size[1]
    # Resize the image
    resized_image = image.resize(desired_size)
    channels = len(resized_image.split())
    # Convert the image to tensor and normalize pixel values
    if channels == 4: 
        resized_image = resized_image.convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_image = transform(resized_image).unsqueeze(0)
    # Convert the image tensor to the appropriate data type
    input_image = input_image.to(torch.float32).to(device)

    from stable_diffusion_guided_generator import StableDiffusionGuidedGenerator
    stable_diffusion_model = StableDiffusionGuidedGenerator(cfg, device)
    RH, RW = H // 8 * 8, W // 8 * 8 # Let's make it a multiple of 8
    rgb_BCHW_HW8 = F.interpolate(
        input_image, (RH, RW), mode="bilinear", align_corners=False
    )
    cond_rgb_BCHW_HW8 = rgb_BCHW_HW8.detach().clone()
    prompts = "a cat", 
    neg_prompt= "unrealistic"
    pretrained_model_name_or_path = cfg["pretrained_model_name_or_path"]
    print(pretrained_model_name_or_path)
    prompt_encoder = StableDiffusionPromptProcessor(pretrained_model_name_or_path, device)
    with torch.no_grad():
        text_embeddings = prompt_encoder.get_text_embeddings(prompts, neg_prompt)
        print(text_embeddings.shape) #[2,77,768]
    output = stable_diffusion_model(rgb_BCHW_HW8, text_embeddings)

