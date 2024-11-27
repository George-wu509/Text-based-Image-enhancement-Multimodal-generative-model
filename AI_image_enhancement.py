import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import os

# ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

python_dir = "D:/Python"  # "C:/other/Python" or "D:/Python"
sys.path.append(os.path.abspath(python_dir + "/sam2"))
from sam2.build_sam import build_sam2_video_predictor

# Development update
"""
1. -> add python libraries and update env note
2. -> restruct code structure based on sd and controlnet ids 
3. -> add sam2 and some image load functions
4. -> add fully seg map functionality
5. -> remove image from ImageEnhancer constructor

"""
# knowledge
"""
 1. transformers 是 Hugging Face 提供的強大 NLP 庫. AutoTokenizer 是一個自動化的工具，用於根據模型名稱或路徑加載適合該模型的分詞器. 
 tokenizer = AutoTokenizer.from_pretrained(), tokenizer是分詞器將文本分解為分詞器的子單元. 
 tokens = tokenizer(text), example: text = "Hello, how are you?"->tokens:['hello', ',', 'how', 'are', 'you', '?']
 ids = tokenizer.encode(text) 將文本轉換為 ID 序列（張量）. example: text = "Hello, how are you?" -> ids=[101, 7592, 1010, 2129, 2024, 2017, 102]
 text = tokenizer.decode(ids) 將 ID 序列轉換回可讀文本。 example: ids=[101, 7592, 1010, 2129, 2024, 2017, 102] -> [CLS] hello, how are you? [SEP]
 * 如果要輸出LLM只需要tokens = tokenizer(text)因為 self.tokenizer(prompt) 返回的結果已經包含了 input_ids，即模型所需的數字化輸入。因此，您不需要額外調用 tokenizer.encode() 再進行轉換。
 * tokenizer.encode(text) 方法比tokenizer(text)包含了更多功能，可以一步完成分詞、轉換為 ID 序列以及添加特殊標記（如 [CLS] 和 [SEP]），適合簡單場景下的快速處理。

2. AutoModelForCausalLM 是一個自動化的工具,用於加載適合因果語言建模(Causal Language Modeling)的預訓練模型。給定序列的前 N 個詞，模型學習預測第 N+1 個詞。例如，輸入 "The sky is"，模型輸出 "blue"
用model = AutoModelForCausalLM.from_pretrained()建立model, 然後用output_ids = model.generate(input_ids)就可以生成output. 
generate的輸入輸出都是ids. text輸入前要用input_ids = encode(text), 輸出的output_ids則用output = decode(output_ids)得到output.
example: text = "The sky is" -> output = "The sky is blue and full of stars"


"""


class LLama3Processor:
    """
    處理器：解析影像增強需求，生成相關的 ControlNet 配置。
    支持通過 Prompt Engineering 和微調增強模型能力。
    """
    def __init__(self, model_name="llama3"):
        """
        初始化 LLama3 模型與分詞器。
        
        參數:
        - model_name (str): 預訓練模型名稱或路徑。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Prompt 模板
        self.prompt_template = """
        你是一個專家級任務解析助手，負責解析用戶的圖像處理需求，並生成多步驟增強計劃。以下是一些常見需求及其對應的配置：

        ### 示例 1
        輸入: "將圖像進行denoise和super-resolution處理"
        輸出:
        [
            {
                "task": "denoise",
                "controlnet": "sd-controlnet-hed",
                "prompt": "a clean, noise-free image, photorealistic, high resolution",
                "negative_prompt": "noisy, low quality, blurry",
                "steps": 50,
                "generator": "random_seed_42"
            },
            {
                "task": "super-resolution",
                "controlnet": "sd-controlnet-normal",
                "prompt": "a high-resolution, detailed image, photorealistic, 8K quality",
                "negative_prompt": "lowres, pixelated, low quality",
                "steps": 50,
                "generator": "random_seed_123"
            }
        ]

        ### 示例 2
        輸入: "移除背景，並將物體A移到圖像中間"
        輸出:
        [
            {
                "task": "remove background",
                "controlnet": "sd-controlnet-seg",
                "prompt": "remove background, clean isolated object, photorealistic, high resolution",
                "negative_prompt": "cluttered background, low quality, poor isolation",
                "steps": 50,
                "generator": "random_seed_42"
            },
            {
                "task": "reposition object",
                "controlnet": "sd-controlnet-seg",
                "prompt": "reposition specified object naturally, seamless, photorealistic",
                "negative_prompt": "unnatural placement, poor alignment, low quality",
                "steps": 60,
                "generator": "random_seed_123"
            }
        ]

        ### 示例 3
        輸入: "改善圖像質量，應用HDR效果，並修正光照"
        輸出:
        [
            {
                "task": "improve image quality",
                "controlnet": "sd-controlnet-seg",
                "prompt": "enhance image quality, vivid colors, sharp focus, photorealistic",
                "negative_prompt": "low quality, blurry, dull colors",
                "steps": 50,
                "generator": "random_seed_42"
            },
            {
                "task": "apply to HDR",
                "controlnet": "sd-controlnet-depth",
                "prompt": "a stunning HDR effect image, vivid colors, high contrast, photorealistic",
                "negative_prompt": "flat, dull colors, low contrast, low quality",
                "steps": 50,
                "generator": "random_seed_123"
            },
            {
                "task": "correct light",
                "controlnet": "sd-controlnet-normal",
                "prompt": "correct light, well-lit image, photorealistic, vivid colors",
                "negative_prompt": "overexposed, underexposed, low quality",
                "steps": 50,
                "generator": "random_seed_456"
            }
        ]

        ### 支持的應用場景與配置
        1. Denoise: 去除噪點並保持圖像細節。
            ControlNet: sd-controlnet-hed
            Prompt: "a clean, noise-free image, photorealistic, high resolution"
            Negative Prompt: "noisy, low quality, blurry"
            Steps: 50

        2. Sharpen: 增強圖像銳度，提升邊緣細節。
            ControlNet: sd-controlnet-normal
            Prompt: "a sharp, crisp, detailed image, photorealistic, high resolution"
            Negative Prompt: "blurry, soft edges, low quality"
            Steps: 50

        3. Image Inpainting: 修復損壞或丟失的圖像區域。
            ControlNet: sd-controlnet-scribble
            Prompt: "fill the missing areas naturally, seamless inpainting, photorealistic"
            Negative Prompt: "unnatural, mismatched, low quality"
            Steps: 60

        4. Dehaze: 移除霧氣，提升圖像清晰度。
            ControlNet: sd-controlnet-depth
            Prompt: "a clear, haze-free image, photorealistic, vibrant colors"
            Negative Prompt: "hazy, foggy, low contrast"
            Steps: 50

        5. Deblur: 去除模糊效果，提升圖像清晰度。
            ControlNet: sd-controlnet-hed
            Prompt: "a sharp, motion-free image, photorealistic, high resolution"
            Negative Prompt: "blurry, motion blur, low quality"
            Steps: 50

        6. Super-Resolution: 增強解析度，適用於 4K 或 8K 圖像。
            ControlNet: sd-controlnet-normal
            Prompt: "a high-resolution, detailed image, photorealistic, 8K quality"
            Negative Prompt: "lowres, pixelated, low quality"
            Steps: 50

        7. Fix White Balance: 修正白平衡，改善色彩準確性。
            ControlNet: sd-controlnet-depth
            Prompt: "correct white balance, natural colors, photorealistic, high resolution"
            Negative Prompt: "unnatural colors, poor white balance, low quality"
            Steps: 50

        8. Apply to HDR: 增強動態範圍，模擬 HDR 效果。
            ControlNet: sd-controlnet-depth
            Prompt: "a stunning HDR effect image, vivid colors, high contrast, photorealistic"
            Negative Prompt: "flat, dull colors, low contrast, low quality"
            Steps: 50

        9. Improve Image Quality: 全面提升圖像質量。
            ControlNet: sd-controlnet-seg
            Prompt: "enhance image quality, vivid colors, sharp focus, photorealistic"
            Negative Prompt: "low quality, blurry, dull colors"
            Steps: 50

        10. Colorize: 為黑白照片添加自然色彩。
            ControlNet: sd-controlnet-depth
            Prompt: "add natural colors to grayscale image, vibrant, photorealistic"
            Negative Prompt: "monochrome, dull colors, low quality"
            Steps: 50

        11. Correct Light: 調整光照，修復過亮或過暗的區域。
            ControlNet: sd-controlnet-normal
            Prompt: "correct light, well-lit image, photorealistic, vivid colors"
            Negative Prompt: "overexposed, underexposed, low quality"
            Steps: 50

        12. Remove Object: 移除指定物體。
            ControlNet: sd-controlnet-seg
            Prompt: "remove specified object from the image, seamless, high resolution"
            Negative Prompt: "low quality, unnatural removal, mismatched"
            Steps: 60

        13. Reposition Object: 移動指定物體到新位置。
            ControlNet: sd-controlnet-seg
            Prompt: "reposition specified object naturally, seamless, photorealistic"
            Negative Prompt: "unnatural placement, poor alignment, low quality"
            Steps: 60

        14. Add Object: 添加指定物體。
            ControlNet: sd-controlnet-seg
            Prompt: "add specified object naturally, seamless blending, photorealistic"
            Negative Prompt: "unnatural addition, mismatched objects, low quality"
            Steps: 60

        15. Remove and Generate Background: 移除現有背景並生成新背景。
            ControlNet: sd-controlnet-seg
            Prompt: "replace background with specified scene, seamless blending, photorealistic"
            Negative Prompt: "unnatural background, mismatched, low quality"
            Steps: 60

        現在解析以下需求，並生成對應的多步驟配置：
        輸入: "{input_text}"
        輸出:
        """

    def generate_prompts_and_modules(self, text_input: str) -> List[Dict]:
        """
        從輸入生成影像增強的配置列表。
        
        參數:
        - text_input (str): 用戶輸入的影像增強需求。
        
        返回:
        - results (list[dict]): 每個影像增強操作的配置列表。
        """
        # 插入輸入到 Prompt 模板
        prompt = self.prompt_template.format(input_text=text_input)
        
        # 編碼輸入
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # 使用模型生成輸出
        outputs = self.model.generate(tokens.input_ids, max_new_tokens=256, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 將生成的文本解析為結構化字典列表
        try:
            results = eval(generated_text)  # 假設模型生成 JSON 格式輸出
        except Exception as e:
            raise ValueError(f"無法解析生成的輸出: {e}")
        
        return results

    def train_model(self, dataset, output_dir="./finetuned_llama3", epochs=3):
        """
        微調 LLama3 模型，增強其處理影像增強需求的能力。
        
        參數:
        - dataset (Dataset): 包含影像增強指令和標籤的數據集（Hugging Face 格式）。
        - output_dir (str): 微調後模型的保存路徑。
        - epochs (int): 訓練輪數。
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=epochs,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
        )

        # 設置 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        # 開始訓練
        trainer.train()
        self.model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

class ImageEnhancer:
    """
    包含多種影像增強功能的類，支持 ControlNet 和專門模型（如 ESRGAN、SAM)。
    """
    def __init__(self, proj_dir):
        """
        初始化 Stable Diffusion、ControlNet 和其他專門模型。

        參數:
        - sd_model_name (str): Stable Diffusion 模型名稱或路徑。
        - controlnet_model_paths (dict): ControlNet 模組路徑字典。
        - sam_model_path (str): Segment Anything Model 的路徑。
        """
        # dir setting
        self.proj_dir = proj_dir
        self.out_dir = os.path.join(proj_dir, "output")
        self.device = self.check_device()

        # stable diffusion and control net
        self.sd_model_name="runwayml/stable-diffusion-v1-5"
        self.controlnet_model_paths={
            "sd-controlnet-hed": "path_to_hed_model",
            "sd-controlnet-depth": "path_to_depth_model",
            "sd-controlnet-normal": "path_to_depth_model",
            "sd-controlnet-scribble": "path_to_depth_model",
            "sd-controlnet-seg": "path_to_depth_model",
        }
        #self.sd_pipeline = StableDiffusionControlNetPipeline.from_pretrained(self.sd_model_name)
        #self.controlnet_models = {
        #    key: ControlNetModel.from_pretrained(path)
        #    for key, path in self.controlnet_model_paths.items()
        #}
        #self.sam = sam_model_registry["default"](checkpoint=sam_model_path)
        #self.sam_predictor = SamPredictor(self.sam)

        # 專門模型（例如 ESRGAN）
        self.other_models = {
            "super_resolution": None,  # 可以加載 Real-ESRGAN 或其他模型
            "deblur": None,  # 加載去模糊專用模型
        }

    def _apply_controlnet(self, image, prompt, negative_prompt, module, steps, generator):
        """
        通用影像增強方法。
        """
        if module not in self.controlnet_models:
            raise ValueError(f"ControlNet module '{module}' not found.")
        
        controlnet = self.controlnet_models[module]
        self.sd_pipeline.controlnet = controlnet

        result = self.sd_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        return result

    # 各影像增強功能
    def enhance_image_canny(self, img_file, prompt, steps):

        # Load images
        canny_image, _ = self.preprocess_canny_image(img_file)
        self.save_image(canny_image, "canny_image.jpg")

        # controlnet and sd model ids
        sd_model_id =  "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-canny"
        
        # stable diffusion with controlnet pipeline 
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)
        pipe = pipe.to(self.device)

        # pipeline scheduler and setting
        # DDIMScheduler：穩定且生成速度較快，適合一般應用, PNDMScheduler：支持類似 PLMS 的方法，生成細節更豐富。DPMSolverMultistepScheduler：加速生成，質量和速度均衡。
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        #pipe.enable_xformers_memory_efficient_attention()

        # prompt and generator 
        prompt = "Sandra Oh, best quality, extremely detailed"
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality"
        generator = torch.manual_seed(2)

        # run pipeline
        # pipe parameters: prompt, image, control_image, negative_prompt, generator, num_inference_steps
        generated_output = pipe(prompt=prompt, image=canny_image, negative_prompt=negative_prompt, generator=generator, num_inference_steps=steps)
        self.save_SDResultimage(generated_output, "canny_output.jpg")

    def enhance_image_openpose(self, img_file, prompt, steps):

        # Load images
        poses, _ = self.preprocess_openpose_image(img_file)
        self.save_image(poses, "poses_image.jpg")

        # controlnet and sd model ids
        sd_model_id =  "runwayml/stable-diffusion-v1-5"
        controlnet_id = "fusing/stable-diffusion-v1-5-controlnet-openpose"
        
        # stable diffusion with controlnet pipeline 
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)

        # pipeline scheduler and setting
        # DDIMScheduler：穩定且生成速度較快，適合一般應用, PNDMScheduler：支持類似 PLMS 的方法，生成細節更豐富。DPMSolverMultistepScheduler：加速生成，質量和速度均衡。
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        #pipe.enable_xformers_memory_efficient_attention()

        # prompt and generator 
        prompt = "super-hero character, best quality, extremely detailed"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        generator = torch.manual_seed(2)

        # run pipeline
        # pipe parameters: prompt, image, control_image, negative_prompt, generator, num_inference_steps
        generated_output = pipe(prompt=prompt, image=poses, negative_prompt=negative_prompt, generator=generator, num_inference_steps=steps)
        self.save_SDResultimage(generated_output, "poses_output.jpg")

    def enhance_image_inpainting(self, img_file, prompt, steps):

        # Load images
        mask, image = self.preprocess_inpainting_image(img_file)
        self.save_image(mask, "inpainting_mask.jpg")

        # controlnet and sd model ids
        sd_model_id =  "runwayml/stable-diffusion-inpainting"
        #controlnet_id = ""
        
        # stable diffusion with controlnet pipeline 
        #controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)

        # pipeline scheduler and setting
        # DDIMScheduler：穩定且生成速度較快，適合一般應用, PNDMScheduler：支持類似 PLMS 的方法，生成細節更豐富。DPMSolverMultistepScheduler：加速生成，質量和速度均衡。
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        #pipe.enable_xformers_memory_efficient_attention()

        # prompt and generator 
        prompt = "A small robot, high resolution, sitting on a park bench"
        negative_prompt = "ugly, blurry, monochrome, lowres, bad anatomy, worst quality, low quality"
        generator = torch.manual_seed(2)

        # run pipeline
        # pipe parameters: prompt, image, control_image, negative_prompt, generator, num_inference_steps
        generated_output = pipe(prompt=prompt, image=image, control_image=mask, negative_prompt=negative_prompt, generator=generator, num_inference_steps=steps)
        self.save_SDResultimage(generated_output, "inpainting_output.jpg")

    def enhance_image_controlnet_inpainting(self, img_file, prompt, steps):

        # Load images
        mask, image = self.preprocess_inpainting_image(img_file)
        self.save_image(mask, "inpainting_controlnet_mask.jpg")

        # controlnet and sd model ids
        sd_model_id =  "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-seg"
        
        # stable diffusion with controlnet pipeline 
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)

        # pipeline scheduler and setting
        # DDIMScheduler：穩定且生成速度較快，適合一般應用, PNDMScheduler：支持類似 PLMS 的方法，生成細節更豐富。DPMSolverMultistepScheduler：加速生成，質量和速度均衡。
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        #pipe.enable_xformers_memory_efficient_attention()

        # prompt and generator 
        prompt = "A small robot, high resolution, sitting on a park bench"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        generator = torch.manual_seed(2)

        # run pipeline
        # pipe parameters: prompt, image, control_image, negative_prompt, generator, num_inference_steps
        generated_output = pipe(prompt=prompt, image=image, control_image=mask, negative_prompt=negative_prompt, generator=generator, num_inference_steps=steps)
        self.save_SDResultimage(generated_output, "inpainting_controlnet_output.jpg")

    def enhance_image_SuperResolution(self, img_file, prompt, steps):

        # Load images
        image, mask = self.process_inpainting(self)
        self.save_image(mask, "inpainting_controlnet_mask.jpg")

        # controlnet and sd model ids
        sd_model_id =  "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-seg"
        
        # stable diffusion with controlnet pipeline 
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)

        # pipeline scheduler and setting
        # DDIMScheduler：穩定且生成速度較快，適合一般應用, PNDMScheduler：支持類似 PLMS 的方法，生成細節更豐富。DPMSolverMultistepScheduler：加速生成，質量和速度均衡。
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        #pipe.enable_xformers_memory_efficient_attention()

        # prompt and generator 
        prompt = "A small robot, high resolution, sitting on a park bench"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        generator = torch.manual_seed(2)

        # run pipeline
        # pipe parameters: prompt, image, control_image, negative_prompt, generator, num_inference_steps
        generated_output = pipe(prompt=prompt, image=image, control_image=mask, negative_prompt=negative_prompt, generator=generator, num_inference_steps=steps)
        self.save_SDResultimage(generated_output, "inpainting_controlnet_output.jpg")

    def enhance_image_ESRGAN(self, img_file):

        # Load image
        input_image, _ = self.preprocess_image(img_file)

        # 使用 RealESRGAN_x4plus 模型，適用於 4 倍超分辨率
        model = RRDBNet(num_in_ch=3, num_out_ch=3, nf=64, nb=23, gc=32, scale=4)  # RRDBNet 模型架構
        model_path = os.path.join(self.proj_dir, "model", "RealESRGAN_x4plus.pth")
        upsampler = RealESRGANer(
            scale=4,  # 放大比例，4 表示 4 倍超分辨率
            model_path=model_path,  # 模型權重文件
            model=model,  # 模型結構
            pre_pad=10,  # 用於避免邊緣效應的預填充像素
            half=True  # 使用半精度以節省內存和加速推理
        )

        # 執行超分辨率
        sr_image, _ = upsampler.enhance(input_image, outscale=4)
        self.save_image(sr_image, "RealEsrgan_output.jpg")

# ------------------------------
    def enhance_image_colorize(self, img_file, steps):
        # Load grayscale image (to be colorized)
        grayscale_image, _ = self.preprocess_grayscale_image(img_file)  # Assume preprocess_grayscale_image exists
        self.save_image(grayscale_image, "grayscale_image.jpg")

        # ControlNet and SD model IDs
        sd_model_id = "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-depth"  # Use depth model for colorization guidance

        # Stable Diffusion with ControlNet pipeline
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        # Pipeline scheduler and setting
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Prompt and generator
        prompt = "a colorful vibrant version of the grayscale input, natural colors, photorealistic, high quality"
        negative_prompt = "monochrome, grayscale, low quality, unrealistic colors, oversaturation"
        generator = torch.manual_seed(42)

        # Run pipeline
        generated_output = pipe(
            prompt=prompt,
            image=grayscale_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps
        )
        self.save_SDResultimage(generated_output, "colorized_output.jpg")

    def enhance_image_hed(self, img_file, steps, mode="denoise"):
        """
        基於 sd-controlnet-hed 進行去噪或去模糊處理。
        mode: 'denoise' or 'deblur'
        """
        # 預處理圖像，生成 HED 邊緣圖
        hed_image, _ = self.preprocess_hed_image(img_file)
        self.save_image(hed_image, "hed_image.jpg")

        # ControlNet 和 SD 模型 ID
        sd_model_id = "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-hed"

        # 加載 ControlNet 模型
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        # 設置 Scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Prompt 設置
        if mode == "denoise":
            prompt = "a clean, noise-free, high-resolution image, photorealistic, highly detailed"
            negative_prompt = "low quality, noisy, blurry, bad composition"
        elif mode == "deblur":
            prompt = "a sharp, clear, highly detailed image, photorealistic, high resolution"
            negative_prompt = "low quality, blurry, motion blur, out of focus"
        else:
            raise ValueError("Invalid mode. Use 'denoise' or 'deblur'.")

        # 生成器 (可控隨機性)
        generator = torch.manual_seed(42)

        # 執行管道生成
        generated_output = pipe(
            prompt=prompt,
            image=hed_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps
        )
        output_path = f"hed_output_{mode}.jpg"
        self.save_SDResultimage(generated_output, output_path)

    def enhance_image_normal(self, img_file, steps, mode="sharpen"):
        """
        基於 sd-controlnet-normal 進行銳化、超解析或光照校正。
        mode: 'sharpen', 'super-resolution', or 'correct-light'
        """
        # 預處理圖像，生成 Normal Map
        normal_image, _ = self.preprocess_normal_image(img_file)
        self.save_image(normal_image, "normal_image.jpg")

        # ControlNet 和 SD 模型 ID
        sd_model_id = "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-normal"

        # 加載 ControlNet 模型
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        # 設置 Scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Prompt 設置
        if mode == "sharpen":
            prompt = "a sharp, highly detailed, vivid image, photorealistic, high resolution"
            negative_prompt = "blurry, low quality, soft edges, out of focus"
        elif mode == "super-resolution":
            prompt = "a high-resolution, highly detailed, crisp image, photorealistic, 8K quality"
            negative_prompt = "lowres, blurry, pixelated, low quality"
        elif mode == "correct-light":
            prompt = "a well-lit, naturally illuminated image, photorealistic, vivid colors, high resolution"
            negative_prompt = "overexposed, underexposed, low quality, dull colors"
        else:
            raise ValueError("Invalid mode. Use 'sharpen', 'super-resolution', or 'correct-light'.")

        # 生成器 (可控隨機性)
        generator = torch.manual_seed(42)

        # 執行管道生成
        generated_output = pipe(
            prompt=prompt,
            image=normal_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps
        )
        output_path = f"normal_output_{mode}.jpg"
        self.save_SDResultimage(generated_output, output_path)

    def enhance_image_depth(self, img_file, steps, mode="dehaze"):
        """
        基於 sd-controlnet-depth 進行除霧或應用 HDR 效果。
        mode: 'dehaze' or 'hdr'
        """
        # 預處理圖像，生成深度圖
        depth_image, _ = self.preprocess_depth_image(img_file)
        self.save_image(depth_image, "depth_image.jpg")

        # ControlNet 和 Stable Diffusion 模型 ID
        sd_model_id = "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-depth"

        # 加載 ControlNet 模型
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        # 設置 Scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Prompt 設置
        if mode == "dehaze":
            prompt = "a clear, crisp image without haze, photorealistic, high resolution, vibrant colors"
            negative_prompt = "hazy, foggy, low contrast, dull colors, low quality"
        elif mode == "hdr":
            prompt = "a stunning HDR effect image with vivid colors, high contrast, photorealistic, highly detailed"
            negative_prompt = "flat, dull colors, low contrast, low resolution, low quality"
        else:
            raise ValueError("Invalid mode. Use 'dehaze' or 'hdr'.")

        # 生成器 (可控隨機性)
        generator = torch.manual_seed(42)

        # 執行管道生成
        generated_output = pipe(
            prompt=prompt,
            image=depth_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps
        )
        output_path = f"depth_output_{mode}.jpg"
        self.save_SDResultimage(generated_output, output_path)

    def enhance_image_seg(self, img_file, steps, mode="improve_quality", extra_args=None):
        """
        基於 sd-controlnet-seg 進行圖像質量改善、背景處理及對象操作。
        mode: 'improve_quality', 'remove_background', 'replace_background',
              'remove_multi_object', 'add_multi_object', 'move_multi_object'
        extra_args: 額外參數，根據 mode 的需求提供，如背景圖像或對象位置。
        """
        # 預處理圖像，生成 Segmentation Mask
        seg_image, _ = self.preprocess_seg_image(img_file)
        self.save_image(seg_image, "seg_image.jpg")

        # ControlNet 和 SD 模型 ID
        sd_model_id = "runwayml/stable-diffusion-v1-5"
        controlnet_id = "lllyasviel/sd-controlnet-seg"

        # 加載 ControlNet 模型
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        # 設置 Scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Prompt 和 Negative Prompt 設置
        if mode == "improve_quality":
            prompt = "a high-quality, photorealistic image with enhanced details, vibrant colors, and sharp focus"
            negative_prompt = "low quality, blurry, dull colors, low resolution"
        elif mode == "remove_background":
            prompt = "an image with no background, clean and isolated object, high resolution"
            negative_prompt = "low quality, cluttered background, noisy image"
        elif mode == "replace_background":
            prompt = f"an image with a new background: {extra_args.get('new_background', 'a beautiful scenery')}, high resolution"
            negative_prompt = "low quality, mismatched background, low resolution"
        elif mode == "remove_multi_object":
            prompt = "an image with specified objects removed, clean, high resolution"
            negative_prompt = "low quality, cluttered image, low resolution"
        elif mode == "add_multi_object":
            prompt = f"an image with added objects: {extra_args.get('new_objects', 'a tree, a car')}, natural blending, high resolution"
            negative_prompt = "low quality, unrealistic objects, poor blending"
        elif mode == "move_multi_object":
            prompt = f"an image with objects moved to new positions: {extra_args.get('object_positions', 'specified locations')}, seamless transition, high resolution"
            negative_prompt = "low quality, unnatural placements, low resolution"
        else:
            raise ValueError("Invalid mode. Choose a valid operation mode.")

        # 生成器 (可控隨機性)
        generator = torch.manual_seed(42)

        # 執行管道生成
        generated_output = pipe(
            prompt=prompt,
            image=seg_image,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps
        )
        output_path = f"seg_output_{mode}.jpg"
        self.save_SDResultimage(generated_output, output_path)


    def preprocess_image(self, img_file):
        img_file = os.path.join(self.proj_dir, img_file)
        img = Image.open(img_file).convert("RGB")

        return low_res_image, img
        
    def preprocess_canny_image(self, img_file):

        img_file = os.path.join(self.proj_dir, img_file)
        img = np.array(Image.open(img_file))
        low_threshold = 100
        high_threshold = 200
        if img.ndim == 3:  # 如果圖像有 3 個維度，表示是彩色圖像
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image = cv2.Canny(img_gray, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        return canny_image, img
    
    def preprocess_openpose_image(self, img_file):
        model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        img_file = os.path.join(self.proj_dir, img_file)
        img = np.array(Image.open(img_file))
        poses = model(img)
        return poses, img

    def preprocess_inpainting_image(self, img_file):
        img_file = os.path.join(self.proj_dir, img_file)
        pass
        #return mask, img

# ------------------------------
    def preprocess_hed_image(self, img_file):
        """
        生成 HED (Holistically-Nested Edge Detection) 圖像作為 ControlNet 的輸入。
        """
        # 加載圖像
        image = cv2.imread(img_file)
        if image is None:
            raise FileNotFoundError(f"圖像文件未找到: {img_file}")
        
        # 將圖像轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用 OpenCV 的邊緣檢測算法 (可選：HED 模型的集成可更準確)
        edges = cv2.Canny(gray, 50, 150)  # 使用 Canny 邊緣檢測

        # 將邊緣圖轉為 3 通道格式 (HED 通常期望彩色輸入)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb, img_file

    def preprocess_normal_image(self, img_file):
        """
        生成 Normal Map 作為 ControlNet 的輸入。
        """
        # 加載圖像
        image = cv2.imread(img_file)
        if image is None:
            raise FileNotFoundError(f"圖像文件未找到: {img_file}")
        
        # 將圖像轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 創建一個假的 normal map 作為輸入 (簡化處理，可替換為深度模型)
        # 使用 Sobel 運算生成 x 和 y 梯度
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 計算法向量的 RGB 值
        magnitude = np.sqrt(grad_x**2 + grad_y**2 + 1)
        normal_map = np.zeros_like(image, dtype=np.float32)
        normal_map[..., 0] = grad_x / magnitude  # R
        normal_map[..., 1] = grad_y / magnitude  # G
        normal_map[..., 2] = 1 / magnitude       # B
        normal_map = ((normal_map + 1) / 2 * 255).astype(np.uint8)

        return normal_map, img_file

    def preprocess_depth_image(self, img_file):
        """
        生成深度圖 (Depth Map) 作為 ControlNet 的輸入。
        """
        # 加載圖像
        image = cv2.imread(img_file)
        if image is None:
            raise FileNotFoundError(f"圖像文件未找到: {img_file}")

        # 將圖像轉為灰度圖 (簡化深度處理)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 假設輸入的深度由灰度值近似，深度越深，灰度值越低 (反向操作)
        depth_map = cv2.equalizeHist(255 - gray)

        # 將灰度圖轉為 3 通道格式 (ControlNet Depth 模型需要彩色輸入)
        depth_map_rgb = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        return depth_map_rgb, img_file

    def preprocess_seg_image(self, img_file):
        """
        使用 SAM 生成 Segmentation Mask。
        """
        # 加載圖像
        image = cv2.imread(img_file)
        if image is None:
            raise FileNotFoundError(f"圖像文件未找到: {img_file}")
        self.predictor.set_image(image)

        # 自動生成 Segmentation Mask
        masks = self.predictor.generate(segment_prompts=None)  # 自動生成所有區域
        combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

        # 將所有 Segmentation Mask 合併
        for mask in masks:
            combined_mask = np.maximum(combined_mask, mask["segmentation"].astype(np.uint8) * 255)

        # 保存並返回 Mask
        seg_mask_rgb = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
        return seg_mask_rgb, img_file


    # 影像處理輔助功能
    def load_image(self, file_path):
        return Image.open(file_path)

    def save_image(self, image, output_file):
        save_path = os.path.join(self.out_dir, output_file)
        image.save(save_path)

    def save_SDResultimage(self, output, output_file):
        # 檢查 `output.images` 的類型
        if isinstance(output.images, list):  # 如果是列表，處理多張圖像
            for idx, image in enumerate(output.images):
                save_path = os.path.join(self.out_dir, output_file)
                #save_path = os.path.join(output_folder, f"output_{idx}.png")
                image.save(save_path)
                print(f"Saved image {idx} to {save_path}")
        elif isinstance(output.images, Image.Image):  # 如果是單張圖像
            save_path = os.path.join(self.out_dir, output_file)
            output.images.save(save_path)
            print(f"Saved single image to {save_path}")
        else:
            print("Error: `output.images` is not a valid image or list of images.")

    def display_image(self, image, title="Image"):
        plt.imshow(np.array(image))
        plt.title(title)
        plt.axis("off")
        plt.show()

    def check_device():
        if torch.cuda.is_available:
            device = "cuda"
        else:
            device = "cpu"
        return device

# Example Usage
if __name__ == "__main__":
    # Initialize the processor and enhancer
    # llama_processor = LLama3Processor(model_name="meta-llama/Llama-2-7b-hf")
    proj_dir = os.path.join(python_dir, "@AI_image_enhancement")
    enhancer = ImageEnhancer(proj_dir)

    # Example user input
    user_input = "please denoise this photo and improve its image quality."

    # Generate prompt and module
    #prompt, module = llama_processor.generate_prompt_and_module(user_input)
    #print(f"Generated Prompt: {prompt}")
    #print(f"Selected Module: {module}")

    # Load an example image (replace with your image loading code)
    #example_image = torch.randn(1, 3, 512, 512)  # Placeholder for actual image
    prompt = ""

    # Enhance the image
    process_steps = 20
    img_file = "image/input_image_vermeer.png"
    enhanced_image = enhancer.enhance_image_canny(img_file, prompt, process_steps)