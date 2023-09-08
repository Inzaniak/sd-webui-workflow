import math

import modules.scripts as scripts
import gradio as gr
from PIL import Image

from modules import processing, shared, images, devices
from modules.processing import Processed, process_images
from modules.shared import opts, state


class Script(scripts.Script):
    def title(self):
        return "Workflow"

    def show(self, is_img2img):
        return is_img2img
    
    def check_orientation(self, img):
        """Check if image is portrait, landscape or square"""
        x,y = img.size
        if x/y > 1.2:
            return 'Horizontal'
        elif y/x > 1.2:
            return 'Vertical'
        else:
            return 'Square'

    def ui(self, is_img2img):
        phase = gr.Radio(label='Phase', choices=["None",'768','1152','1920'], value="None")
        orientation = gr.Radio(label='Orientation', choices=["Guess","Horizontal", "Vertical", "Square"], value="Guess")
        force_denoising = gr.Checkbox(label='Force denoising', value=False, elem_id=self.elem_id("force_denoising"))
        denoising = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Denoising strength', value=0.5)
        with gr.Accordion("Performance", open = False):
            with gr.Group():
                phase_1_nr = gr.Number(label='Phase 1 Nr', value=4, min_value=1, step=1)
                phase_2_nr = gr.Number(label='Phase 2 Nr', value=4, min_value=1, step=1)
                phase_3_nr = gr.Number(label='Phase 3 Nr', value=1, min_value=1, step=1)
            with gr.Group():
                with gr.Group():
                    phase_1_x = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 1 X', value=512)
                    phase_1_y = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 1 Y', value=768)
                with gr.Group():
                    phase_2_x = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 2 X', value=768)
                    phase_2_y = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 2 Y', value=1152)
                with gr.Group():
                    phase_3_x = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 3 X', value=1280)
                    phase_3_y = gr.Slider(minimum=0, maximum=1920, step=2, label='Phase 3 Y', value=1920)
        return [phase, force_denoising, denoising, orientation, phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y]

    def run(self, p, phase, force_denoising, denoising, orientation, phase_1_nr, phase_2_nr, phase_3_nr, phase_1_x, phase_1_y, phase_2_x, phase_2_y, phase_3_x, phase_3_y):
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
            orientation = self.check_orientation(init_image)
            
        if orientation == "Horizontal":
            p.width, p.height = p.height, p.width
        elif orientation == "Square":
            p.width = p.height = max(p.width, p.height)
        processed = process_images(p)
        return processed
