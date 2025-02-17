from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionInpaintPipeline
import torch
from typing import Optional
from PIL import Image
from controlnet_aux import MLSDdetector
import io
from starlette.responses import StreamingResponse
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
#from mobile_sam import sam_model_registry, SamPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class InteriorDesignGenerator:
    def __init__(self, huggingface_token: str):
        self.token = huggingface_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Define the device attribute
        self.setup_pipelines()
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    def setup_pipelines(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Stable Diffusion Pipeline
        stable_diffusion_model = "runwayml/stable-diffusion-v1-5"
        self.stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(stable_diffusion_model).to(device)

        # Load ControlNet Model and Pipeline
        controlnet_model_path = "lllyasviel/sd-controlnet-mlsd"
        self.controlnet_model = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
        self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            stable_diffusion_model, 
            controlnet=self.controlnet_model, 
            safety_checker=None, 
            torch_dtype=torch.float16
        ).to(device)
        self.controlnet_pipeline.scheduler = UniPCMultistepScheduler.from_config(self.controlnet_pipeline.scheduler.config)
        # Enable memory-efficient attention if xformers is installed
        #self.controlnet_pipeline.enable_xformers_memory_efficient_attention()
        #self.controlnet_pipeline.enable_model_cpu_offload()

        # Use the device attribute throughout the setup_pipelines method
        self.stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to(self.device)

        self.controlnet_model = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", 
            torch_dtype=torch.float16
        )
        self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=self.controlnet_model, 
            safety_checker=None, 
            torch_dtype=torch.float16
        ).to(self.device)

        self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16
        ).to(self.device)

        # Initialize SAM model
        sam_model_type = "vit_h"
        sam_checkpoint = "./sam_vit_h_4b8939.pth"
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam) 
        self.sam.eval()

        #sam_checkpoint = "MobileSAM/weights/mobile_sam.pt"
        #model_type = "vit_t"
        #self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        #self.sam.to(self.device)
        #self.sam.eval()

    def generate_mask(self, image: Image.Image, bbox: np.ndarray) -> Image.Image:
        try:
            # Convert PIL Image to numpy array
            np_image = np.array(image)

            # Use SAM to generate mask
            predictor = SamPredictor(self.sam)
            predictor.set_image(np_image)

            # Get mask for the bounding box
            masks, scores, _ = predictor.predict(
                box=bbox,  # Pass the bounding box
                multimask_output=True,
            )

            # Select the mask with the highest score
            best_mask_index = np.argmax(scores)
            best_mask = masks[best_mask_index]

            mask_image = Image.fromarray(best_mask.astype(np.uint8) * 255)
            return mask_image

        except Exception as e:
            logger.error(f"Error in generate_mask: {e}", exc_info=True)
            raise

    def segment_objects(self, image: Image.Image, bbox: np.ndarray) -> Image.Image:
        try:
            # Convert the original PIL image to a numpy array
            image_np = np.array(image.convert("RGB"))

            # Generate the binary mask using the provided bounding box
            mask_image = self.generate_mask(image, bbox)
            mask_np = np.array(mask_image)

            # Ensure the mask is boolean
            mask_bool = mask_np > 128

            # Initialize an output array with zeros, with the same shape as the input image
            output_np = np.zeros_like(image_np)

            # Apply the mask to the original image to copy the segmented part
            for c in range(3):  # For each color channel
                output_np[:, :, c] = image_np[:, :, c] * mask_bool

            # Convert the numpy array back to a PIL Image
            segmented_image = Image.fromarray(output_np, 'RGB')
            return segmented_image

        except Exception as e:
            logger.error(f"Error in segment_objects: {e}", exc_info=True)
            raise


    def generate_design(self, concept: str, control_image: Optional[Image.Image] = None) -> Image.Image:
        print("Starting generate_design method...")
        try:
            if control_image:
                print("Processing control image with MLSDdetector...")
                # Process control image with MLSDdetector
                control_image = self.mlsd(control_image)
                print("Generating design with control image...")
                image = self.controlnet_pipeline(prompt=concept, image=control_image, num_inference_steps=20).images[0]
            else:
                print("Generating design without control image...")
                image = self.stable_diffusion_pipeline(concept).images[0]
            print("Design generation completed.")
            return image
        except Exception as e:
            print(f"Error during design generation: {e}")
            raise
    
    def generate_inpainting(self, prompt: str, image: Image.Image, mask_image: Image.Image) -> Image.Image:
        return self.inpainting_pipeline(prompt=prompt, image=image, mask_image=mask_image).images[0]

# Instantiate the generator with your Hugging Face API token
generator = InteriorDesignGenerator("")

@app.post("/generate-design/")
async def generate_design(concept: str = Form(...), control_image: Optional[UploadFile] = File(None)):
    try:
        print(f"Received concept: {concept}, control_image: {control_image}")
        if control_image:
            print("Reading control image...")
            control_image_data = Image.open(io.BytesIO(await control_image.read()))
            print("Control image read successfully.")
        else:
            control_image_data = None
            print("No control image provided.")

        print("Calling generate_design method...")
        image = generator.generate_design(concept, control_image=control_image_data)

        print("Design generation successful. Preparing response...")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        print("Response prepared. Sending back to client.")
        # Save the inpainted image
        generated_image_filename = "generated_image.png"
        image.save(generated_image_filename)
        logger.info(f"Saved generated image to {generated_image_filename}")
        return StreamingResponse(io.BytesIO(img_byte_arr.getvalue()), media_type="image/png")
    except Exception as e:
        print(f"Error in generate_design endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-mask/")
async def generate_mask(original_image: UploadFile = File(...), bbox: str = Form(...)):
    try:
        logger.info(f"Received bbox: {bbox}")
        image_data = Image.open(io.BytesIO(await original_image.read()))

        # Convert the string representation of the bbox to a numpy array
        bbox_array = np.array(eval(bbox))

        # Check if bbox_array is of shape (2,2)
        if bbox_array.shape != (2, 2):
            raise ValueError("Bounding box should be of shape (2,2)")

        # Generate the mask using the provided image and bbox
        mask_image = generator.generate_mask(image_data, bbox_array)

        # Convert the mask image to a byte array for response
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(io.BytesIO(img_byte_arr.getvalue()), media_type="image/png")
    except Exception as e:
        logger.error(f"Error in generate_mask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
 
@app.post("/segment-with-box/")
async def segment_with_box(image: UploadFile = File(...), bbox: str = Form(...)):
    try:
        # Parse the bounding box string into a numpy array
        bbox = np.fromstring(bbox.strip(']['), sep=',', dtype=int)
        if bbox.shape[0] != 4:
            raise ValueError("Bounding box should contain four values [x1, y1, x2, y2]")

        # Read the image
        image_data = Image.open(io.BytesIO(await image.read()))
        np_image = np.array(image_data.convert('RGB'))

        # Set the image in the SAM predictor
        generator.sam_predictor.set_image(np_image)

        # Predict the mask for the bounding box
        box = np.array([bbox])  # Reshape bbox to match expected input [x1, y1, x2, y2]
        masks, scores, _ = generator.sam_predictor.predict(box=box, multimask_output=True)

        # Select the mask with the highest score
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]

        # Apply the mask to the image
        segmented_np_image = apply_mask(np_image, best_mask)

        # Convert the numpy array back to a PIL image
        segmented_image = Image.fromarray(segmented_np_image)

        # Prepare the response
        img_byte_arr = io.BytesIO()
        segmented_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(io.BytesIO(img_byte_arr.getvalue()), media_type='image/png')
    except Exception as e:
        logger.error(f"Error in segment_with_box: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def apply_mask(image_np, mask):
    """Apply the binary mask to the image."""
    mask_bool = mask > 0.5
    output_np = np.zeros_like(image_np)
    for c in range(3):  # Apply mask for each color channel
        output_np[:, :, c] = image_np[:, :, c] * mask_bool
    return output_np

  
@app.post("/inpaint-image/")
async def inpaint_image(prompt: str = Form(...), original_image: UploadFile = File(...), mask_image: UploadFile = File(...)):
    try:
        # Read the original image and mask image data
        logger.info("Reading original and mask image data...")
        original_image_data = await original_image.read()
        mask_image_data = await mask_image.read()
        
        # Open the images using PIL
        logger.info("Opening images using PIL...")
        image_data = Image.open(io.BytesIO(original_image_data))
        mask_image_data = Image.open(io.BytesIO(mask_image_data))

        # Log the size of the images
        logger.info(f"Original image size: {image_data.size}")
        logger.info(f"Mask image size: {mask_image_data.size}")

        # Save the original and mask images to disk
        original_image_filename = "original_image.png"
        mask_image_filename = "mask_image.png"
        image_data.save(original_image_filename)
        mask_image_data.save(mask_image_filename)
        logger.info(f"Saved original image to {original_image_filename}")
        logger.info(f"Saved mask image to {mask_image_filename}")

        # Generate the inpainting
        logger.info("Generating inpainting...")
        inpainted_image = generator.generate_inpainting(prompt, image_data, mask_image_data)
        # Resize the inpainted image to match the original image's size
        inpainted_image = inpainted_image.resize(image_data.size)


        # Save the inpainted image
        inpainted_image_filename = "inpainted_image.png"
        inpainted_image.save(inpainted_image_filename)
        logger.info(f"Saved inpainted image to {inpainted_image_filename}")
        
        # Log success of inpainting generation
        logger.info("Inpainting generated successfully.")

        # Save the inpainted image to a byte array
        logger.info("Saving inpainted image to byte array...")
        img_byte_arr = io.BytesIO()
        inpainted_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return the inpainted image as a streaming response
        logger.info("Returning the inpainted image as a streaming response...")
        return StreamingResponse(io.BytesIO(img_byte_arr.getvalue()), media_type="image/png")
    except Exception as e:
        # Log any errors that occur
        logger.error(f"Error in inpaint_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
