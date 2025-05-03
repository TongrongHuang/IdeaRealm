import gc
import os
import random
import re
import threading
import time
from datetime import datetime, timedelta

import gradio as gr
import torch
import numpy as np
from diffusers import (CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline,
                       CogVideoXPipeline, CogView4Pipeline)
from diffusers.models import CogView4Transformer2DModel
from diffusers.utils import export_to_video, load_image
from moviepy import VideoFileClip
from torchao.quantization import int8_weight_only, quantize_
from transformers import GlmModel
from PIL import Image
from zhipuai import ZhipuAI

device = "cuda" if torch.cuda.is_available() else "cpu"

CogView4_path = './models/CogView4-6B'
GLM_path = None
CogVideoX_T2V_path = './models/CogVideoX1.5-5B'
CogVideoX_I2V_path = './models/CogVideoX1.5-5B-I2V'

loras = [
    ("无具体风格", "./models/IdeaRealm_CogView4_LoRAs/9_lora.safetensors"),
    ("侘寂风", "./models/IdeaRealm_CogView4_LoRAs/1_lora.safetensors"),
    ("公寓", "./models/IdeaRealm_CogView4_LoRAs/2_lora.safetensors"),
    ("奶油风", "./models/IdeaRealm_CogView4_LoRAs/3_lora.safetensors"),
    ("日式", "./models/IdeaRealm_CogView4_LoRAs/4_lora.safetensors"),
    ("沙发背景墙", "./models/IdeaRealm_CogView4_LoRAs/5_lora.safetensors"),
    ("书店", "./models/IdeaRealm_CogView4_LoRAs/6_lora.safetensors"),
    ("网咖", "./models/IdeaRealm_CogView4_LoRAs/7_lora.safetensors"),
    ("卧室", "./models/IdeaRealm_CogView4_LoRAs/8_lora.safetensors"),
    ("无主灯", "./models/IdeaRealm_CogView4_LoRAs/10_lora.safetensors"),
    ("现代简约", "./models/IdeaRealm_CogView4_LoRAs/11_lora.safetensors"),
    ("新中式", "./models/IdeaRealm_CogView4_LoRAs/12_lora.safetensors"),
    ("早教中心", "./models/IdeaRealm_CogView4_LoRAs/13_lora.safetensors"),
    ("中古风", "./models/IdeaRealm_CogView4_LoRAs/14_lora.safetensors"),
    ("轻奢", "./models/IdeaRealm_CogView4_LoRAs/15_lora.safetensors"),
    ("北欧", "./models/IdeaRealm_CogView4_LoRAs/16_lora.safetensors"),
    ("美式", "./models/IdeaRealm_CogView4_LoRAs/17_lora.safetensors"),
    ("法式", "./models/IdeaRealm_CogView4_LoRAs/18_lora.safetensors"),
    ("别墅", "./models/IdeaRealm_CogView4_LoRAs/19_lora.safetensors"),
    ("工业风", "./models/IdeaRealm_CogView4_LoRAs/20_lora.safetensors"),
    ("原木风", "./models/IdeaRealm_CogView4_LoRAs/21_lora.safetensors"),
    ("书房", "./models/IdeaRealm_CogView4_LoRAs/22_lora.safetensors"),
    ("高级灰", "./models/IdeaRealm_CogView4_LoRAs/23_lora.safetensors"),
    ("台式", "./models/IdeaRealm_CogView4_LoRAs/24_lora.safetensors"),
    ("复式", "./models/IdeaRealm_CogView4_LoRAs/25_lora.safetensors"),
    ("小户型", "./models/IdeaRealm_CogView4_LoRAs/26_lora.safetensors"),
]

def load_model(task, reduce_gpu=True):

    # 加载文生图模型
    if task == "text_to_image":
        # text_encoder = GlmModel.from_pretrained(GLM_path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        # transformer = CogView4Transformer2DModel.from_pretrained(GLM_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        # quantize_(text_encoder, int8_weight_only())
        # quantize_(transformer, int8_weight_only())
        pipe = CogView4Pipeline.from_pretrained(
            CogView4_path,
            # text_encoder=text_encoder,
            # transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to(device)

    # 加载文生视频模型
    if task == "text_to_video":
        pipe = CogVideoXPipeline.from_pretrained(CogVideoX_T2V_path, torch_dtype=torch.bfloat16).to(device)

    # 加载图生视频模型
    if task == "image_to_video":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(CogVideoX_I2V_path, torch_dtype=torch.bfloat16).to(device)

    # 降低GPU显存使用率
    if reduce_gpu:
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    return pipe

os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

def clean_string(s):
    s = s.replace("\n", " ")
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

def cogview4_convert_prompt(
    prompt: str,
    key: str,
    retry_times: int = 3,
) -> str:
    os.environ["OPENAI_API_KEY"] = key
    if not key:
        return prompt

    client = ZhipuAI(api_key=key)
    prompt = clean_string(prompt)
    for i in range(retry_times):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": 'You are a bilingual image description assistant that works with an image generation bot.  You work with an assistant bot that will draw anything you say . \n    For example ,For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" or "阳光透过树林的美丽清晨" will trigger your partner bot to output an image of a forest morning, as described . \n    You will be prompted by people looking to create detailed , amazing images . The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive . \n    There are a few rules to follow : \n    - Input can be in Chinese or English. If input is in English, prompt should be written in English. If input is in Chinese, prompt should be written in Chinese.\n    - You will only ever output a single image description per user request .\n    - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting. \n    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.\n    - Do not provide the process and explanation, just return the modified description . \n    ',
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "An anime girl stands amidst a dense flower bush."',
                    },
                    {
                        "role": "assistant",
                        "content": "This image is a beautifully crafted digital illustration in an anime style. It features Suzuran, a character from the video game Arknights. She stands gracefully amidst a picturesque meadow with lush green grass and scattered wildflowers that gently sway in the breeze. Her attire includes a detailed outfit with layered ruffles and intricate fastenings, reflecting both elegance and functionality typical of the game's aesthetic. Suzuran holds a small bouquet of flowers delicately in her hands, adding to the serene atmosphere.\n\nProminent fox ears extend from her head, signifying her animal spirit form or persona within the game. The background showcases rolling hills covered in dense foliage under a brilliant blue sky dotted with fluffy white clouds. Golden leaves float whimsically through the air, enhancing the magical quality of the scene. A distant building partially obscured by vegetation hints at human presence without detracting from the natural beauty surrounding Suzuran. Overall, this artwork captures a moment of tranquility and charm, embodying the essence of the character and the enchanting world she inhabits.",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "Draw a bright convertible car with a sense of artistic design."',
                    },
                    {
                        "role": "assistant",
                        "content": 'The image showcases a meticulously crafted Delahaye 135 M.S. Special Roadster from 1937 in a highly polished and realistic rendering style that highlights its luxurious design and impeccable details. The car\'s body is an elegant deep brown with a glossy finish, exuding sophistication and timeless beauty. Its aerodynamic, streamlined shape features smooth curves accentuated by chrome detailing on the fenders and running boards. The front grille has three prominent circular headlights, adding to its classic charm while the dual exhaust pipes are tastefully integrated into the rear fenders.\n\nThe open cockpit area reveals sumptuous tan leather seats, emphasizing both comfort and elegance for the driver and passenger. The vehicle’s number plate reads "MB 175". Photographed against a dark gradient background, the focus remains solely on this automotive masterpiece, allowing viewers to appreciate every intricate detail without distraction. This image captures not just a car but a piece of art, reflecting the luxury and craftsmanship emblematic of pre-war European automobiles.',
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "画一个白发、神情忧郁的动漫女孩，手里拿着一支香烟"',
                    },
                    {
                        "role": "assistant",
                        "content": "这幅图像是一幅动漫风格的插画，画中描绘了一位长发飘逸的白发女孩。她神情忧郁，双眼低垂，脸上带着微微的愁容。女孩穿着浅色外套，里面搭配深色衬衫和领带，增添了她沉静却时尚的外表。她的手靠近嘴边，似乎在拿着一支香烟，画作的说明也表示她在吸烟。背景描绘了一个冬季的城市场景，地面和建筑物上覆盖着积雪。几株稀疏的植物从雪地中探出，街边停着一辆车，似乎是空荡荡的街道，增强了场景的荒凉氛围。这幅作品整体风格细致精美，典型的现代动漫插画风格，既捕捉了角色的情感表达，也呈现了周围环境的静谧氛围",
                    },
                    {
                        "role": "user",
                        "content": 'Create an imaginative image descriptive caption for the user input : "一张红色的海报，中间写有“开门大吉”"',
                    },
                    {
                        "role": "assistant",
                        "content": "这张图片采用了海报风格，色彩鲜艳，主要以红色和金色为主，寓意吉祥如意。在画面的正中间是一块红色的大匾，上面用白色大字写着“开门大吉”，四角点缀着金色的装饰图案，显得格外喜庆。匾额上方悬挂着一对红色的灯笼，增添了节日的气氛。背景左右两侧是传统的绿色中式建筑，屋顶呈现出典型的飞檐翘角设计。底部有祥云朵朵和可爱的卡通福袋，象征着好运福气满满。整张海报传达出浓厚的节日氛围。",
                    },
                    {
                        "role": "user",
                        "content": f"Create an imaginative image descriptive caption for the user input : {prompt}",
                    },
                ],
                model="glm-4-plus",
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=256,
            )
            prompt = response.choices[0].message.content
            if prompt:
                prompt = clean_string(prompt)
                break
        except Exception:
            pass

    return prompt

def cogvideox_convert_prompt(
    prompt: str,
    key: str,
    retry_times: int = 3
) -> str:
    os.environ["OPENAI_API_KEY"] = key
    if not key:
        return prompt

    client = ZhipuAI(api_key=key)
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            model="glm-4-plus",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=256,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt

def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path

def cogview4_infer(
    prompt: str,
    randomize_seed: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    num_images: int,
    lora: str,
    progress=gr.Progress(track_tqdm=True),
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    pipe = load_model(
        task="text_to_image",
        reduce_gpu=True
    )

    if lora != "无具体风格" or lora != "./models/IdeaRealm_CogView4_LoRAs/9_lora.safetensors":
        pipe.load_lora_weights(lora)

    images = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        generator=torch.Generator().manual_seed(randomize_seed),

    ).images

    return images, randomize_seed

def cogvideox_t2v_infer(
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    num_videos_per_prompt: int,
    height: int,
    width: int,
    randomize_seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    torch.cuda.empty_cache()

    pipe = load_model(
        task="text_to_video",
        reduce_gpu=True
    )

    video_tensor = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(randomize_seed),
    ).frames[0]

    video = save_video(video_tensor)

    return video, randomize_seed

def cogvideox_i2v_infer(
    prompt: str,
    image: str,
    num_videos_per_prompt: int,
    num_inference_steps: int,
    num_frames: int,
    guidance_scale: float,
    randomize_seed: int,
    height: int,
    width: int,
    progress=gr.Progress(track_tqdm=True),
):
    torch.cuda.empty_cache()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    pipe = load_model(
        task="image_to_video",
        reduce_gpu=True
    )

    video_tensor = pipe(
        prompt=prompt,
        image=image,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=torch.Generator(device="cuda").manual_seed(randomize_seed),
    ).frames[0]

    video = save_video(video_tensor)

    return video, randomize_seed

def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path

def convert_to_gif(video_path):
    clip = VideoFileClip(video_path)
    clip = clip.with_fps(8)
    clip = clip.resized(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path

def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)

threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks(fill_width=True, fill_height=True) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            筑意绘界 IdeaRealm
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=7):
            with gr.Tabs(selected=0) as tabs:
                with gr.TabItem("Text to Image", id=0):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                t2i_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=5)
                            with gr.Row():
                                gr.Markdown(
                                    "✨按下增强提示按钮后，将使用 GLM-4 模型来润色提示并覆盖原始提示。"
                                )
                                with gr.Column():
                                    t2i_key = gr.Textbox(label="API密钥",  placeholder="在此输入API密钥", type="password", max_lines=1)
                                    t2i_enhance_button = gr.Button("✨ 提示词润色(可选)")
                            with gr.Row():
                                t2i_lora_choice = gr.Radio(choices=loras, label="选择室内风格")
                            with gr.Row():
                                t2i_num_inference_steps = gr.Slider(1, 100, step=1, label="Inference Steps", value=50, interactive=True)
                                t2i_guidance_scale = gr.Slider(1, 100, step=1, label="Guidance Scale", value=6.0, interactive=True)
                            with gr.Row():
                                t2i_height = gr.Slider(1, 1024, step=1, label="Height", value=512, interactive=True)
                                t2i_width = gr.Slider(1, 1024, step=1, label="Width", value=512, interactive=True)
                            with gr.Row():
                                t2i_num_images = gr.Number(label="一次性生成图片数量",minimum=1, maximum=8, value=4, interactive=True)
                                t2i_seed = gr.Number(label="随机种子", value=random.randint(1, 65536), interactive=True)
                            with gr.Row():
                                t2i_random_button = gr.Button("🎲 随机种子生成")
                            with gr.Row():
                                t2i_enable_scale = gr.Checkbox(label="超分辨率 (720 × 480 -> 2880 × 1920)", value=False)
                                t2i_reduce_gpu = gr.Checkbox(label="使用CPU加载模型", value=False)
                            with gr.Row():
                                t2i_generate_image_button = gr.Button("🖼 生成图片")
                        with gr.Column():
                            t2i_image_output = gr.Gallery(label="图片生成区", height=650, interactive=False, show_label=True, show_download_button=True)
                            with gr.Row():
                                send_to_i2v_button = gr.Button("发送至图生视频作为输入图像")
                    with gr.Row():
                        # 预先准备的样例数据（图片路径 + 提示词）
                        t2i_example_dir = "/root/script/sources/image"  # 替换为你的图片目录
                        t2i_examples = [
                            ["This image depicts a modern, minimalist bedroom with a neutral color palette. The room features a large bed with a wooden headboard, dressed in light-colored bedding and a brown throw blanket. Above the bed hangs a large abstract painting with green and beige tones. To the left of the bed is a small round side table with a bowl, and a pendant light hangs above it. On the right side, there is a built-in wooden desk with a brown chair, and a small vase with branches is placed on the desk. The walls are white, and the floor is covered with a light-colored rug. The overall design is clean, simple, and elegant, emphasizing functionality and comfort.", os.path.join(t2i_example_dir, "t2i-1.jpg")],
                            ["This image depicts a modern, minimalist living space with an open-plan layout. The room features a neutral color palette dominated by white walls and light wooden flooring. On the left, there is a built-in white shelving unit with a television mounted above it. The center of the room showcases a dining area with a wooden table and chairs, complemented by two pendant lights hanging above. To the right, there is a comfortable white sofa adorned with cushions, and in front of it, a small wooden coffee table with a glass and a potted plant. The kitchen area in the background includes white cabinetry and a refrigerator, maintaining the clean and cohesive design. The space is enhanced with green plants, adding a touch of nature and freshness to the interior.", os.path.join(t2i_example_dir, "t2i-2.jpg")],
                            ["This image depicts a modern living room with a blend of traditional and contemporary elements. The room features a large, ornate chandelier hanging from a white ceiling, providing warm lighting. The focal point is a wall-mounted flat-screen TV set against a backdrop of a golden wallpaper adorned with a floral and bird design, adding an artistic touch. Below the TV is a wooden media console with a matching coffee table in the center of the room, which holds decorative items. The flooring is light wood, complemented by a patterned area rug. To the right, there is a large window with curtains, offering a view of the outdoors and allowing natural light to enter. Adjacent to the window is a small laundry area with a washing machine and storage cabinets. The room is furnished with comfortable seating, including a sofa and armchairs, creating a cozy and inviting atmosphere.", os.path.join(t2i_example_dir, "t2i-3.jpg")]
                        ]
                        gr.Markdown("## 🖼️ 曾经生成的优秀作品")
                        # 只展示不交互的样例区
                    with gr.Row():
                        gr.Examples(
                            examples=t2i_examples,
                            inputs=[gr.Textbox(label="对应提示词", visible=False),
                                    gr.Image(label="生成作品", type="filepath", visible=False)],
                            label="历史作品库",  # 自定义标题
                            examples_per_page=3  # 每页显示数量
                        )

                with gr.TabItem("Text to Video", id=1):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                t2v_prompt = gr.Textbox(label="Prompt(不超过200个单词)", placeholder="在此输入提示词", lines=5)
                            with gr.Row():
                                gr.Markdown(
                                    "✨按下增强提示按钮后，将使用 GLM-4 模型来润色提示并覆盖原始提示。"
                                )
                                with gr.Column():
                                    t2v_key = gr.Textbox(label="API密钥", placeholder="在此输入API密钥",
                                                     type="password", max_lines=1)
                                    t2v_enhance_button = gr.Button("✨ 提示词润色(可选)")
                            with gr.Row():
                                t2v_num_inference_steps = gr.Slider(1, 100, step=1, label="Inference Steps", value=50, interactive=True)
                                t2v_guidance_scale = gr.Slider(1, 100, step=1, label="Guidance Scale", value=6.0, interactive=True)
                            with gr.Row():
                                t2v_height = gr.Slider(1, 1024, step=1, label="Height", value=480, interactive=True)
                                t2v_width = gr.Slider(1, 1024, step=1, label="Width", value=720, interactive=True)
                            with gr.Row():
                                i=t2v_num_videos_per_prompt = gr.Number(label="生成视频数", value=1, interactive=True)
                                t2v_seed = gr.Number(label="随机种子", value=random.randint(1, 65536), interactive=True)
                            with gr.Row():
                                t2v_random_button = gr.Button("🎲 随机种子生成")
                            with gr.Row():
                                t2v_enable_scale = gr.Checkbox(label="超分辨率 (720 × 480 -> 2880 × 1920)", value=False)
                                t2v_enable_rife = gr.Checkbox(label="插帧 (8fps -> 16fps)", value=False)
                                t2v_reduce_gpu = gr.Checkbox(label="使用CPU加载模型", value=False)
                            with gr.Row():
                                t2v_generate_image_button = gr.Button("🎬 生成视频")
                        with gr.Column():
                            t2v_video_output = gr.Video("视频生成区", width=720, height=480, interactive=False)
                            with gr.Row():
                                t2v_download_video_button = gr.Button("📥 Download Video")
                                t2v_download_gif_button = gr.Button("📥 Download GIF")
                    with gr.Row():
                        # 预先准备的样例数据（图片路径 + 提示词）
                        t2v_example_dir = "/root/script/sources/video"  # 替换为你的图片目录
                        examples = [
                            ["Generate a high - quality interior design video showcasing a modern living room. The room is bathed in soft, warm natural light streaming through a large floor - to - ceiling window on the right wall. The window features sheer white curtains that gently sway, filtering the sunlight. Outside the window, there’s a serene garden with blooming roses, neatly trimmed hedges, and a small stone pathway. The layout is open - concept, with a plush, L - shaped gray sofa facing a sleek, white marble coffee table. A minimalist fireplace with a black metal frame anchors one wall, above which hangs a large abstract painting. Overhead, a contemporary chandelier with adjustable LED lights provides additional illumination. The video should start with a slow pan from the window, moving across the room, then a smooth tilt down to showcase the geometric - patterned rug. As the camera glides around, it should highlight the built - in wooden shelving filled with books and decorative items, before ending on a close - up of a cozy reading nook with a floor lamp and a single armchair. ", os.path.join(t2v_example_dir, "t2v-1.mp4")],
                            ["Create a video of a rustic - themed dining room interior design. Soft, diffused light filters in through small, rectangular windows with wooden shutters on the far wall, casting gentle shadows. Outside, a charming countryside scene unfolds: rolling green hills dotted with grazing sheep and a winding dirt road leading to a distant farmhouse. The layout centers around a long, reclaimed - wood dining table surrounded by eight ladder - back chairs upholstered in beige fabric. Above the table, a vintage - style wrought - iron chandelier with Edison bulbs hangs, adding a warm glow. Against one wall, a rustic sideboard with brass handles holds fine china and candles. The video should begin with a steady zoom from the window view into the room, then a slow tracking shot around the table, highlighting the intricate wood grain and table setting. It should then smoothly pivot to show the sideboard and the artful arrangement of decor items, before ending with a slow pull - out to capture the entire room.", os.path.join(t2v_example_dir, "t2v-2.mp4")],
                            ["Generate a video for a contemporary bedroom interior design. The room is illuminated by a combination of natural light from a large, arched window on the left wall and warm, ambient LED strip lights. The window offers a view of a bustling cityscape at dusk, with skyscrapers lit up and the orange glow of the setting sun in the background. The layout features a king - sized bed with a tufted, dark gray upholstered headboard positioned against the central wall. On either side, there are sleek, white nightstands with built - in wireless charging stations. A floor - to - ceiling wardrobe with mirrored doors stands adjacent to the window, reflecting the city view. The video starts with a slow crane shot from the ceiling, descending towards the bed. Then, it makes a smooth orbit around the room, focusing on the details of the furniture, the texture of the bedding, and the play of light on the mirrors. Finally, it lingers on the window, showing the contrast between the calm interior and the vibrant city outside.", os.path.join(t2v_example_dir, "t2v-3.mp4")]
                        ]
                        gr.Markdown("## 🖼️ 曾经生成的优秀作品")
                        # 只展示不交互的样例区
                    with gr.Row():
                        gr.Examples(
                            examples=examples,
                            inputs=[gr.Textbox(label="对应提示词", visible=False),
                                    gr.Video(label="生成作品", format="mp4", visible=False, interactive=False, include_audio=True)],
                            label="历史作品库",  # 自定义标题
                            examples_per_page=3  # 每页显示数量
                        )

                with gr.TabItem("Image to Video", id=2):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                with gr.Column():
                                    i2v_prompt = gr.Textbox(label="Prompt(不超过200个单词)", placeholder="在此输入提示词", lines=11)
                                with gr.Column():
                                    i2v_input_image = gr.Image(label="输入图片", interactive=True)
                            with gr.Row():
                                gr.Markdown(
                                    "✨按下增强提示按钮后，将使用 GLM-4 模型来润色提示并覆盖原始提示。"
                                )
                                with gr.Column():
                                    i2v_key = gr.Textbox(label="API密钥", placeholder="在此输入API密钥",
                                                         type="password", max_lines=1)
                                    i2v_enhance_button = gr.Button("✨ 提示词润色(可选)")
                            with gr.Row():
                                i2v_num_inference_steps = gr.Slider(1, 100, step=1, label="Inference Steps", value=50, interactive=True)
                                i2v_guidance_scale = gr.Slider(1, 100, step=1, label="Guidance Scale", value=6.0, interactive=True)
                            with gr.Row():
                                i2v_height = gr.Slider(1, 1024, step=1, label="Height", value=480, interactive=True)
                                i2v_width = gr.Slider(1, 1024, step=1, label="Width", value=720, interactive=True)
                            with gr.Row():
                                i2v_num_frames = gr.Slider(1, 200, step=1, label="视频帧数", value=48, interactive=True)
                                with gr.Column():
                                    gr.Markdown(
                                        "要生成的帧数必须能被 self.vae_scale_factor_temporal 整除。生成的视频将包含 1 个额外帧，因为 CogVideoX 以 (num_seconds * fps + 1) 帧为条件，其中 num_seconds 为 6，fps 为 8。但是，由于视频可以以任意 fps 保存，因此唯一需要满足的条件就是上面提到的可整除性。"
                                        "建议默认48帧即可。"
                                    )
                            with gr.Row():
                                i2v_num_videos_per_prompt = gr.Number(label="生成视数", value=1, interactive=True)
                                i2v_seed = gr.Number(label="随机种子", value=random.randint(1, 65536), interactive=True)
                            with gr.Row():
                                i2v_random_button = gr.Button("🎲 随机种子生成")
                            with gr.Row():
                                i2v_enable_scale = gr.Checkbox(label="超分辨率 (720 × 480 -> 2880 × 1920)", value=False)
                                i2v_enable_rife = gr.Checkbox(label="插帧 (8fps -> 16fps)", value=False)
                                i2v_reduce_gpu = gr.Checkbox(label="使用CPU加载模型", value=False)
                            with gr.Row():
                                i2v_generate_image_button = gr.Button("🎬 生成视频")
                        with gr.Column():
                            i2v_video_output = gr.Video("视频生成区", width=720, height=480, interactive=False)
                            with gr.Row():
                                i2v_download_video_button = gr.Button("📥 Download Video")
                                i2v_download_gif_button = gr.Button("📥 Download GIF")
                    with gr.Row():
                        # 预先准备的样例数据（图片路径 + 提示词）
                        i2v_image_dir = "/root/script/sources/image"
                        i2v_video_dir = "/root/script/sources/video"  # 替换为你的图片目录
                        i2v_examples = [
                            ["Create a video showcasing this modern living space with smooth and fluid camera movements. Start with a slow - paced dolly shot entering the room from the doorway in the background, gradually focusing on the light - filled interior. As the camera moves forward, it gently pans to the right, highlighting the sleek, gray sofa adorned with soft cushions. Then, execute a slow tilt - down to capture the elegant coffee table with its gold - accented legs and the lush green plant on top. Next, the camera glides to the right, focusing on the round dining table with the vase of fresh flowers and the open book. Though there are no visible windows in the image, if we were to imagine one, it could be on the far wall, with a view of a tranquil street lined with trees and modern apartment buildings. Finish with a slow crane shot that ascends from the dining area, providing a sweeping view of the entire room, emphasizing the clean lines and minimalist decor. ", os.path.join(i2v_image_dir, "i2v-1.jpg"), os.path.join(i2v_video_dir, "i2v-1.mp4")],
                            ["Create a video that showcases this cozy and minimalist dining room with smooth, flowing camera movements. Begin with a slow fade - in as the camera gently moves forward from the doorway area, focusing on the warm wooden textures of the table and chairs. The two woven pendant lights above the table should be highlighted with a brief upward tilt of the camera. Then, the camera should smoothly pan to the left to reveal the sleek wooden cabinet, emphasizing its clean lines and natural finish. After that, it should glide towards the window with sheer curtains. As the camera approaches, imagine a soft, out - of - focus view of a serene garden outside through the curtains, with blooming flowers and gently swaying foliage. Finally, the camera should execute a slow orbit around the dining table, capturing different angles of the decor elements like the small vase with branches and the bowl, before ending with a wide - angle shot that encompasses the entire room, highlighting its harmonious and inviting atmosphere. ", os.path.join(i2v_image_dir, "i2v-2.jpg"), os.path.join(i2v_video_dir, "i2v-2.mp4")],
                            ["""Create a video that showcases this modern, tech - inspired bedroom with smooth and dynamic camera movements. Start with a slow - moving tracking shot that enters the room from the doorway, gradually revealing the sleek, dark - toned interior. The camera should then glide to the left, highlighting the gaming setup with high - end chairs and multiple monitors displaying vibrant visuals. Next, execute a gentle tilt - up to focus on the large "Avatar: The Way of Water" poster on the wall, emphasizing the room's entertainment theme. After that, the camera should smoothly pan to the right, showcasing the neatly made bed with its contrasting white and dark bedding. As the camera approaches the window area (though not visible in the image, we can imagine it), it could reveal a view of a bustling cityscape at night, with illuminated skyscrapers and the glow of streetlights. Finally, the camera should perform a slow orbit """, os.path.join(i2v_image_dir, "i2v-3.jpg"), os.path.join(i2v_video_dir, "i2v-3.mp4")]
                        ]
                        gr.Markdown("## 🖼️ 曾经生成的优秀作品")
                        # 只展示不交互的样例区
                    with gr.Row():
                        gr.Examples(
                            examples=i2v_examples,
                            inputs=[gr.Textbox(label="对应提示词", visible=False),
                                    gr.Image(label="输入图像", type="filepath", visible=False),
                                    gr.Video(label="生成视频", format="mp4", interactive=False, visible=False, include_audio=True)],
                            label="历史作品库",  # 自定义标题
                            examples_per_page=3  # 每页显示数量
                        )

    # 随机种子生成按钮
    def generate_random_seed():
        return random.randint(0, 65536)
    t2i_random_button.click(fn=generate_random_seed, inputs=None, outputs=[t2i_seed])
    t2v_random_button.click(fn=generate_random_seed, inputs=None, outputs=[t2v_seed])
    i2v_random_button.click(fn=generate_random_seed, inputs=None, outputs=[i2v_seed])

    # 提示词润色
    t2i_enhance_button.click(cogview4_convert_prompt, inputs=[t2i_prompt, t2i_key], outputs=[t2i_prompt])
    t2v_enhance_button.click(cogvideox_convert_prompt, inputs=[t2v_prompt, t2v_key], outputs=[t2v_prompt])
    i2v_enhance_button.click(cogvideox_convert_prompt, inputs=[i2v_prompt, i2v_key], outputs=[i2v_prompt])

    # 生成按钮
    t2i_generate_image_button.click(
        fn=cogview4_infer,
        inputs=[t2i_prompt, t2i_seed, t2i_width, t2i_height, t2i_guidance_scale, t2i_num_inference_steps, t2i_num_images, t2i_lora_choice],
        outputs=[t2i_image_output, t2i_seed]
    )
    t2v_generate_image_button.click(
        fn=cogvideox_t2v_infer,
        inputs=[t2v_prompt, t2v_num_inference_steps, t2v_guidance_scale, t2v_num_videos_per_prompt, t2v_height, t2v_width, t2v_seed],
        outputs=[t2v_video_output, t2v_seed]
    )
    i2v_generate_image_button.click(
        fn=cogvideox_i2v_infer,
        inputs=[i2v_prompt, i2v_input_image, i2v_num_videos_per_prompt, i2v_num_inference_steps, i2v_num_frames, i2v_guidance_scale, i2v_seed, i2v_height, i2v_width],
        outputs=[i2v_video_output, i2v_seed]
    )

    # 其它按钮
    def send_image_to_i2v(image):
        if not image:
            print("无图片可发送,请检查.")
        return image
    send_to_i2v_button.click(
        fn=send_image_to_i2v,
        inputs=[t2i_image_output],
        outputs=[i2v_input_image]
    )

if __name__ == "__main__":
    demo.launch()


