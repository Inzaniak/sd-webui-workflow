import math
import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageChops
import numpy as np
import random
import os
from PIL import ImageFilter
import json

extensions_root = scripts.basedir()
user_dir = os.path.join(extensions_root, 'user')
user_settings_dir = os.path.join(user_dir, 'settings')
user_overlay_dir = os.path.join(user_dir, 'overlays')
os.makedirs(user_settings_dir, exist_ok=True)
os.makedirs(user_overlay_dir, exist_ok=True)


def add_noise_f(img, noise, noise_color=(255, 0, 0), image_mask: Image = None):
    # img: PIL image
    # noise: float, percentage of noise
    # noise_color: tuple, color of the noise in RGB
    # image_mask: PIL image, white areas indicate where to apply noise
    # return: PIL image
    img = img.convert("RGB")
    img = np.array(img)
    noise = int(noise * img.size // img.shape[2])  # Adjust for color channels

    if image_mask is not None:
        image_mask = np.array(image_mask.convert("1"))  # Convert to binary mask

    for _ in range(noise):
        while True:
            x = random.randint(0, img.shape[0] - 1)
            y = random.randint(0, img.shape[1] - 1)
            if image_mask is None or image_mask[x][y]:  # If no mask or pixel in mask is white
                img[x][y][0] = noise_color[0]  # Red channel
                img[x][y][1] = noise_color[1]  # Green channel
                img[x][y][2] = noise_color[2]  # Blue channel
                break  # Exit the loop once a suitable pixel is found

    return Image.fromarray(img)


def swap_pixels_f(image, distance, image_mask: Image.Image = None):
    image = image.convert("RGB")
    if image_mask is not None:
        image_mask = np.array(image_mask.convert("1"))
    width, height = image.size
    new_image = image.copy()
    for x in range(width):
        for y in range(height):
            if image_mask is None or image_mask[y][x]:  # Apply effect if no mask or pixel in mask is white
                r, g, b = image.getpixel((x, y))
                new_x = x + random.randint(-distance, distance)
                new_y = y + random.randint(-distance, distance)
                if new_x < 0:
                    new_x = 0
                if new_x >= width:
                    new_x = width - 1
                if new_y < 0:
                    new_y = 0
                if new_y >= height:
                    new_y = height - 1
                new_image.putpixel((new_x, new_y), (r, g, b, 255))
    return new_image


def add_chromatic_aberration_f(img, shift_amount, image_mask=None):
    img = img.convert("RGB")
    original_img = img.copy()
    r, g, b = img.split()
    r = r.transform(r.size, Image.AFFINE, (1, 0, shift_amount, 0, 1, 0))
    b = b.transform(b.size, Image.AFFINE, (1, 0, -shift_amount, 0, 1, 0))
    img = Image.merge('RGB', (r, g, b))
    if image_mask is not None:
        img.paste(original_img, mask=ImageChops.invert(image_mask.convert("1")))
    return img


def add_overlay_f(base_img, overlay_image_path, alpha, method="blend", image_mask=None):
    base_img = base_img.convert("RGB")
    overlay_img = Image.open(overlay_image_path) if isinstance(overlay_image_path, str) else overlay_image_path
    overlay_img = overlay_img.convert("RGB")
    overlay_img = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)
    if method == "blend":
        combined_img = Image.blend(base_img, overlay_img, alpha)
    elif method == "multiply":
        combined_img = ImageChops.multiply(base_img, overlay_img)
    elif method == "screen":
        combined_img = ImageChops.screen(base_img, overlay_img)
    elif method == "add":
        combined_img = ImageChops.add(base_img, overlay_img)
    elif method == "subtract":
        combined_img = ImageChops.subtract(base_img, overlay_img)
    elif method == "overlay":
        combined_img = ImageChops.overlay(base_img, overlay_img)
    else:
        raise ValueError("Invalid blend method")

    if image_mask is not None:
        combined_img.paste(base_img, mask=ImageChops.invert(image_mask.convert("1")))

    return combined_img


def blur_image_f(img: Image.Image, radius, blur_type="gaussian", image_mask=None):
    # img: PIL image
    # radius: float, blur radius
    # blur_type: str, type of blur ("gaussian", "box", "median")
    # image_mask: PIL image, white areas indicate where to apply blur
    # return: PIL image
    if blur_type == "gaussian":
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    elif blur_type == "box":
        blurred_img = img.filter(ImageFilter.BoxBlur(radius))
    elif blur_type == "median":
        radius = min([3, 5, 9], key=lambda x: abs(x - radius))
        blurred_img = img.filter(ImageFilter.MedianFilter(size=radius))
    else:
        raise ValueError("Invalid blur type")

    if image_mask is not None:
        blurred_img.paste(img, mask=ImageChops.invert(image_mask.convert("1")))

    return blurred_img


def sharpen_image_f(img, radius, percent, threshold, image_mask=None):
    # img: PIL image
    # radius: float, sharpening radius
    # percent: float, sharpening percent
    # threshold: float, sharpening threshold
    # image_mask: PIL image, white areas indicate where to apply sharpening
    # return: PIL image
    sharpened_img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    if image_mask is not None:
        sharpened_img.paste(img, mask=ImageChops.invert(image_mask.convert("1")))

    return sharpened_img


def hide_if_true(in_flag):
    return gr.Checkbox.update(visible=not in_flag)


def hide_if_false(in_flag):
    return gr.Checkbox.update(visible=in_flag)


def hide_if_not_blend(in_flag):
    return gr.Checkbox.update(visible=in_flag == "blend")


def hide_if_not_custom(in_flag):
    return gr.Checkbox.update(visible=in_flag == "Custom")


def list_img_files(path):
    """List all image files in a folder"""
    return [file for file in os.listdir(path) if file.rpartition('.')[2].lower() in ['png', 'jpg', 'jpeg', 'webp']]


def refresh_overlay_choices():
    return gr.Dropdown.update(choices=list_img_files(user_overlay_dir))


def calculate_distance(tuple_1, tuple_2):
    # calculate the distance between two tuples (x,y,z)
    return math.sqrt((tuple_1[0] - tuple_2[0]) ** 2 + (tuple_1[1] - tuple_2[1]) ** 2 + (tuple_1[2] - tuple_2[2]) ** 2)


def save_settings(phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y, phase_1_denoising, phase_2_denoising, phase_3_denoising, save_name):
    settings_dict = {"phases": [
        {"x": int(phase_1_x), "y": int(phase_1_y), "batch": int(phase_1_nr), "denoising": float(phase_1_denoising)},
        {"x": int(phase_2_x), "y": int(phase_2_y), "batch": int(phase_2_nr), "denoising": float(phase_2_denoising)},
        {"x": int(phase_3_x), "y": int(phase_3_y), "batch": int(phase_3_nr), "denoising": float(phase_3_denoising)},
    ]}
    with open(os.path.join(user_settings_dir, f'{save_name}.json'), 'w', encoding='utf-8') as outfile:
        json.dump(settings_dict, outfile, ensure_ascii=False, indent=4)


def load_settings(save_name):
    phases = json.load(open(os.path.join(user_settings_dir, f'{save_name}.json')))['phases']
    return phases[0]["batch"], phases[1]["batch"], phases[2]["batch"], phases[0]["x"], phases[0]["y"], phases[1]["x"], phases[1]["y"], phases[2]["x"], phases[2]["y"], phases[0]["denoising"], phases[1]["denoising"], phases[2]["denoising"]


def check_orientation(img):
    """Check if image is portrait, landscape or square"""
    x, y = img.size
    if x / y > 1.2:
        return 'Horizontal'
    elif y / x > 1.2:
        return 'Vertical'
    else:
        return 'Square'


class Script(scripts.Script):
    image_mask = None
    fx_preview = None
    settings = None

    def title(self):
        return "Workflow"

    def show(self, is_img2img):
        if is_img2img:
            return scripts.AlwaysVisible

    def init_settings(self):
        # initialize settings
        self.settings = {"phases": [{"x": 512, "y": 768, "batch": 6, "denoising": 0.5},
                                    {"x": 768, "y": 1152, "batch": 4, "denoising": 0.5},
                                    {"x": 1280, "y": 1920, "batch": 1, "denoising": 0.2}]}
        try:
            default_settings = os.path.join(user_settings_dir, 'default.json')
            if os.path.isfile(default_settings):
                self.settings = json.load(open(default_settings))
        except Exception as e:
            print(f"Error loading settings: {e}")

    def ui(self, is_img2img):
        self.init_settings()
        with gr.Group():
            with gr.Accordion("Workflow", open=False):
                phase = gr.Radio(label='Phase', choices=["None", '768', '1152', '1920'], value="None")
                orientation = gr.Radio(label='Orientation', choices=["Guess", "Horizontal", "Vertical", "Square"], value="Guess")
                ratio = gr.Radio(label='Ratio', choices=["Base", "2:1"], value="Base")
                force_denoising = gr.Checkbox(label='Force denoising', value=False, elem_id=self.elem_id("force_denoising"))
                denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoising strength', value=0.5)
                with gr.Accordion("Extra", open=False):
                    with gr.Group():
                        enable_extras = gr.Checkbox(label='Enable extras', value=False)
                        with gr.Accordion("Overlay", open=False):
                            # Overlay
                            add_overlay = gr.Checkbox(label='Add overlay')
                            overlay_img = gr.Image(type="pil")
                            overlay_image_path = gr.Dropdown(label='Overlay image', choices=list_img_files(user_overlay_dir))
                            method_select = gr.Radio(label='Overlay method', choices=["blend", "multiply", "screen", "add", "subtract", "overlay"], value="blend")
                            overlay_opacity = gr.Slider(minimum=0.05, maximum=1.0, step=0.05, label='Overlay opacity', value=0.5)
                            add_overlay.change(refresh_overlay_choices, [], overlay_image_path)
                            method_select.change(hide_if_not_blend, method_select, overlay_opacity)
                        with gr.Accordion("Chromatic Aberration", open=False):
                            # Chromatic aberration
                            chromatic_aberration = gr.Checkbox(label='Chromatic aberration')
                            shift_amount = gr.Slider(minimum=1, maximum=100, step=1, label='Shift amount', value=1)
                        with gr.Accordion("Noise", open=False):
                            # Noise
                            add_noise = gr.Checkbox(label='Add noise')
                            noise_amount = gr.Slider(minimum=0.001, maximum=1.0, step=0.001, label='Noise Amount', value=0.010)
                            noise_color = gr.ColorPicker(label='Noise color', value='#ffffff')
                        with gr.Accordion("Swap Pixels", open=False):
                            # Swap
                            swap_pixels = gr.Checkbox(label='Swap pixels')
                            swap_distance = gr.Slider(minimum=1, maximum=100, step=1, label='Swap distance', value=1)
                        with gr.Accordion("Flip", open=False):
                            # Flip
                            flip_vertical = gr.Checkbox(label='Flip vertical')
                            flip_horizontal = gr.Checkbox(label='Flip horizontal')
                        with gr.Accordion("Blur", open=False):
                            # Blur
                            blur_type = gr.Dropdown(label='Blur type', choices=["None", "gaussian", "box", "median"], value="None")
                            blur_radius = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, label='Blur radius', value=1.0)
                        with gr.Accordion("Sharpen", open=False):
                            # Sharpen
                            sharpen = gr.Checkbox(label='Sharpen')
                            sharpen_radius = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, label='Sharpen radius', value=2.0)
                            sharpen_percent = gr.Slider(minimum=0, maximum=300, step=1, label='Sharpen percent', value=150)
                            sharpen_threshold = gr.Slider(minimum=1, maximum=255, step=1, label='Sharpen threshold', value=3)
                        fx_preview = gr.Checkbox(label='Preview FX', value=False)
                with gr.Accordion("Mask", open=False):
                    choose_custom_mask = gr.Radio(label='Choose custom mask', choices=["Default", "White", "Black", "Custom"], value="Default")
                    choose_custom_mask_threshold = gr.Slider(minimum=0, maximum=255, step=1, label='Mask threshold', value=5)
                    choose_custom_mask_color = gr.ColorPicker(label='Custom mask color', value='#ffffff')
                    invert_mask = gr.Checkbox(label='Invert mask')
                    return_mask = gr.Checkbox(label='Return mask')
                    use_only_fx = gr.Checkbox(label='Use only for FX', value=False)

                with gr.Accordion("Performance", open=False):
                    with gr.Accordion("Batch Size", open=False):
                        phase_1_nr = gr.Number(label='Phase 1 Nr', value=self.settings["phases"][0]["batch"], min_value=1, step=1)
                        phase_2_nr = gr.Number(label='Phase 2 Nr', value=self.settings["phases"][1]["batch"], min_value=1, step=1)
                        phase_3_nr = gr.Number(label='Phase 3 Nr', value=self.settings["phases"][2]["batch"], min_value=1, step=1)
                    with gr.Accordion("Width/Height", open=False):
                        phase_1_x = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 1 X', value=self.settings["phases"][0]["x"])
                        phase_1_y = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 1 Y', value=self.settings["phases"][0]["y"])
                        phase_2_x = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 2 X', value=self.settings["phases"][1]["x"])
                        phase_2_y = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 2 Y', value=self.settings["phases"][1]["y"])
                        phase_3_x = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 3 X', value=self.settings["phases"][2]["x"])
                        phase_3_y = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 3 Y', value=self.settings["phases"][2]["y"])
                    with gr.Accordion("Denoising Strengths", open=False):
                        phase_1_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Phase 1 Denoising', value=self.settings["phases"][0]["denoising"])
                        phase_2_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Phase 2 Denoising', value=self.settings["phases"][1]["denoising"])
                        phase_3_denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Phase 3 Denoising', value=self.settings["phases"][2]["denoising"])
                    # save button
                    file_name = gr.Textbox(label="File Name", lines=1, value="default")
                    save_button = gr.Button(value="Save settings", type="button")
                    load_button = gr.Button(value="Load settings", type="button")
                    save_button.click(save_settings, [phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y, phase_1_denoising, phase_2_denoising, phase_3_denoising, file_name], [])
                    load_button.click(load_settings, [file_name], [phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y, phase_1_denoising, phase_2_denoising, phase_3_denoising])
        return [phase, force_denoising, denoising, orientation, phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y, ratio, enable_extras, add_noise, noise_amount, noise_color, fx_preview, swap_pixels, swap_distance, chromatic_aberration, shift_amount, add_overlay, overlay_image_path, choose_custom_mask, choose_custom_mask_threshold, choose_custom_mask_color, overlay_opacity, method_select, return_mask, use_only_fx, overlay_img, invert_mask, flip_vertical, flip_horizontal, blur_type, blur_radius, sharpen, sharpen_percent, sharpen_radius, sharpen_threshold]

    def before_process(self, p, phase, force_denoising, denoising, orientation, phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y, ratio, enable_extras, add_noise, noise_amount, noise_color, fx_preview, swap_pixels, swap_distance, chromatic_aberration, shift_amount, add_overlay, overlay_image_path, choose_custom_mask, choose_custom_mask_threshold, choose_custom_mask_color, overlay_opacity, method_select, return_mask, use_only_fx, overlay_img, invert_mask, flip_vertical, flip_horizontal, blur_type, blur_radius, sharpen, sharpen_percent, sharpen_radius, sharpen_threshold):
        if phase != "None":
            if phase == "768":
                p.width = int(phase_1_x)
                p.height = int(phase_1_y)
                p.batch_size = int(phase_1_nr)
                p.denoising_strength = 0.5
            elif phase == "1152":
                p.width = int(phase_2_x)
                p.height = int(phase_2_y)
                p.batch_size = int(phase_2_nr)
                p.denoising_strength = 0.5
            elif phase == "1920":
                p.width = int(phase_3_x)
                p.height = int(phase_3_y)
                p.batch_size = int(phase_3_nr)
                p.denoising_strength = 0.2

            if force_denoising:
                p.denoising_strength = denoising

            init_image = p.init_images[0]
            if orientation == "Guess":
                orientation = check_orientation(init_image)

            if ratio == "2:1":
                p.width = p.height // 2

            if orientation == "Horizontal":
                p.width, p.height = p.height, p.width
            elif orientation == "Square":
                p.width = p.height = max(p.width, p.height)

            image_mask = p.image_mask

            if choose_custom_mask == "White":
                image_mask = p.init_images[0].convert('L').point(lambda x: 0 if x < 255 - choose_custom_mask_threshold else 255, '1')
            elif choose_custom_mask == "Black":
                image_mask = p.init_images[0].convert('L').point(lambda x: 255 if x < choose_custom_mask_threshold else 0, '1')
            elif choose_custom_mask == "Custom":
                if choose_custom_mask_color.startswith('#'):
                    choose_custom_mask_color = tuple(int(choose_custom_mask_color[i:i + 2], 16) for i in (1, 3, 5))
                else:
                    choose_custom_mask_color = eval(choose_custom_mask_color)
                print(choose_custom_mask_color)
                image_mask = Image.new('1', init_image.size)
                width, height = init_image.size
                init_image = init_image.convert('RGB')
                for x in range(width):
                    for y in range(height):
                        r, g, b = init_image.getpixel((x, y))
                        if calculate_distance((r, g, b), choose_custom_mask_color) < choose_custom_mask_threshold:
                            image_mask.putpixel((x, y), 255)
                        else:
                            image_mask.putpixel((x, y), 0)

            if p.image_mask is not None and image_mask is not None:
                # if there are two image mask merge them using a logical AND
                image_mask = ImageChops.multiply(p.image_mask, image_mask)
                print("Merging masks")

            self.image_mask = image_mask

            if (p.inpainting_mask_invert and p.image_mask) or invert_mask:
                image_mask = ImageChops.invert(image_mask)

            if not use_only_fx:
                p.image_mask = image_mask

            if enable_extras:
                if add_overlay:
                    if overlay_img is None:
                        p.init_images[0] = add_overlay_f(p.init_images[0], os.path.join(user_overlay_dir, overlay_image_path), overlay_opacity, method_select, image_mask)
                    else:
                        p.init_images[0] = add_overlay_f(p.init_images[0], overlay_img, overlay_opacity, method_select, image_mask)

                if chromatic_aberration:
                    if image_mask is not None:
                        p.init_images[0] = add_chromatic_aberration_f(p.init_images[0], shift_amount, image_mask)
                    else:
                        p.init_images[0] = add_chromatic_aberration_f(p.init_images[0], shift_amount)

                if add_noise:
                    # check if noise_color is a hex
                    if noise_color.startswith('#'):
                        noise_color = tuple(int(noise_color[i:i + 2], 16) for i in (1, 3, 5))
                    else:
                        # convert noise color from string e.g.(255,0,0) to tuple
                        noise_color = eval(noise_color)

                    p.init_images[0] = add_noise_f(p.init_images[0], noise_amount, noise_color, image_mask)

                if swap_pixels:
                    p.init_images[0] = swap_pixels_f(p.init_images[0], swap_distance, image_mask)

                if flip_vertical:
                    p.init_images[0] = p.init_images[0].transpose(Image.FLIP_TOP_BOTTOM)

                if flip_horizontal:
                    p.init_images[0] = p.init_images[0].transpose(Image.FLIP_LEFT_RIGHT)

                if blur_type != "None":
                    p.init_images[0] = blur_image_f(p.init_images[0], blur_radius, blur_type, image_mask)

                if sharpen:
                    p.init_images[0] = sharpen_image_f(p.init_images[0], sharpen_radius, sharpen_percent, sharpen_threshold, image_mask)

                if fx_preview:
                    self.fx_preview = p.init_images[0]

    def postprocess(self, p, processed, phase, force_denoising, denoising, orientation, phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y, ratio, enable_extras, add_noise, noise_amount, noise_color, fx_preview, swap_pixels, swap_distance, chromatic_aberration, shift_amount, add_overlay, overlay_image_path, choose_custom_mask, choose_custom_mask_threshold, choose_custom_mask_color, overlay_opacity, method_select, return_mask, use_only_fx, overlay_img, invert_mask, flip_vertical, flip_horizontal, blur_type, blur_radius, sharpen, sharpen_percent, sharpen_radius, sharpen_threshold):
        if phase != "None":
            if return_mask and choose_custom_mask != "Default":
                processed.images.append(self.image_mask)
            if enable_extras:
                if fx_preview:
                    processed.images.append(self.fx_preview)
