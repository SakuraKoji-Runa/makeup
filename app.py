import gradio as gr
import os
import subprocess
from PIL import Image

# 创建缓存文件夹
cache_folder = "image_cache"
os.makedirs(cache_folder, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = 'image_cache'


def predict(input_img):
    # 从输入文件路径中提取文件名
    input_filename = os.path.basename(input_img)
    # 构建保存路径
    save_dir = "output/result/added_prediction"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, input_filename)
    #执行预测脚本
    command = f"python work/PaddleSeg/tools/predict.py \
        --config work/pp_liteseg_CelebAMask-HQ_512x512_30k.yml \
        --model_path Model/model.pdparams \
        --image_path {input_img} \
        --save_dir output/result"

    subprocess.run(command, shell=True)
    print(save_path)
    return save_path


def makeup(input_img, labels, color0, color1, color2, color3, color4, color5):
    # 从输入文件路径中提取文件名
    input_filename, input_extension = os.path.splitext(os.path.basename(input_img))
    # 构建保存路径
    save_dir = "output/makeup"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, input_filename)
    pseudo_path = os.path.join("output/result/pseudo_color_prediction", input_filename + ".png")  # 更改文件名后缀为 .png

    # 生成部位参数
    input_labels = [labels]

    # 处理每个部位列表
    processed_labels = [' '.join(f"{label}" for label in group) for group in input_labels]

    # 输出结果
    parts = ' '.join(processed_labels)

    # 打印合并后的标签列表
    print("合并后的标签列表:", parts)

    # 合并三个颜色列表
    all_colors = [color0, color1, color2, color3, color4,color5]

    # 打印合并后的颜色列表
    print("合并后的颜色列表:", all_colors)
    all_colors = [color for color in all_colors if color is not None]

    # 对每个颜色进行处理
    processed_colors = []

    for color in all_colors:
        # 去除十六进制颜色代码中的 # 符号
        hex_color = color.lstrip('#')
        # 分别提取 R、G、B 分量
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        rgb = [r, g, b]
        rgb_str = ','.join(str(c) for c in rgb)
        colors = f"[{rgb_str}]"
        processed_colors.append(colors)
        colors = ' '.join(processed_colors)
        print("处理后的颜色:", colors)
    # 运行变妆脚本
    makeup_command = f"python work/PaddleSeg/makeup.py \
        --img_path {input_img} \
        --pseudo_path {pseudo_path} \
        --save_dir {save_dir} \
        --parts {parts} \
        --colors {colors}"

    subprocess.run(makeup_command, shell=True)
    print(save_path + "_makeup.png")
    return save_path + "_makeup.png"



theme = gr.themes.Default()

with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        inputs = gr.Image(sources=["upload", "webcam", "clipboard"], type='filepath', label="上传图像")
        outputs = gr.Image(type="pil", label="结果图像", show_download_button=False)
    with gr.Row():
        clear = gr.ClearButton([inputs, outputs])
        submit = gr.Button("submit", variant="primary")
    with gr.Column():
        with gr.Group():
            mask_labels = gr.CheckboxGroup(['发色','脸', '上嘴唇', '下嘴唇', '牙齿', '鼻子'],
                                           label="分割标签",
                                           info="选择变妆部位"
                                           )
            select_colors0 = gr.ColorPicker(label="发色")
            select_colors1 = gr.ColorPicker(label="脸")
            select_colors2 = gr.ColorPicker(label="上嘴唇")
            select_colors3 = gr.ColorPicker(label="下嘴唇")
            select_colors4 = gr.ColorPicker(label="牙齿")
            select_colors5 = gr.ColorPicker(label="鼻子")

    makeup_button = gr.Button("makeup", variant="primary")
    submit.click(fn=predict, inputs=inputs, outputs=outputs)
    makeup_button.click(fn=makeup,
                        inputs=[inputs, mask_labels, select_colors0, select_colors1, select_colors2, select_colors3,
                                select_colors4,select_colors5], outputs=outputs)

demo.launch()
