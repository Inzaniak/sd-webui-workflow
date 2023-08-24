# Workflow
![Alt text](pics\image.png)

## Installation
Just copy the script from the scripts folder above into the extensions folder of your 1111automatic installation. Then restart 1111automatic, by clicking the "Reload UI" button on the bottom of the page.
To run the extension just scroll down in the img2img panel to the script selection and select Workflow.

## Features
*You'll probably need to read this to understand my workflow:* https://medium.com/p/6be78130eb9e

I've created this extension to help me streamline my creation workflow. With this extension I can create pictures using 512x768, 768x1152 and 1280x1920 resolutions without having to change multiple parameters. Read the How to use section for more info.

## How to use
![Alt text](pics/screen.png)
To use the extension you must be in the img2img panel. Then scroll down to the script selection and select Workflow. Now you can select a phase:
- **None**: This is the default phase. It bypasses the extension and lets you use img2img as usual.
- **768**: This will create 4 512x768 pictures with a denoising strength of 0.5. This is useful for inpainting.
- **1152**: This will create 4 768x1152 pictures with a denoising strength of 0.5. This is useful for detailing
- **1920**: This will create 1 1280x1920 picture with a denoising strength of 0.2. This is useful for finalizing.

You also have 2 more parameters:
- **Horizontal**: This will create horizontal pictures instead of vertical ones. It keeps the same logic and resolution of the phases.
- **Force Denoising**: This forces a different denoising strength. This is useful if you want to use a different denoising strength than the one specified in the phase.

The final section is related to managing the performance of the tool. The default values have been tested using an RTX 4080. If you have a less powerful GPU you'll probably need to change these. **You can also change these if you want a different resolution than the ones specified in the phases.**

---
## Made by Inzaniak
![Alt text](pics/logo.png) 


If you'd like to support my work feel free to check out my Patreon: https://www.patreon.com/Inzaniak

Also check my other links:
- **Personal Website**: https://inzaniak.github.io 
- **Deviant Art**: https://www.deviantart.com/inzaniak
- **CivitAI**: https://civitai.com/user/Inzaniak/models