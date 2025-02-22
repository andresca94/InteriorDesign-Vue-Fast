# InteriorDesign-Vue-Fast

ControlNet Stable Diffusion for Interior Design

Design a tranquil Scandinavian living space where simplicity and comfort converge. Picture a soft woolen herringbone throw casually draped over a nearby oak armchair, with a series of potted succulents in clean-lined concrete planters arranged on the floor beside the couch. Above the couch, imagine a gallery wall of monochromatic landscapes in slim black frames, promoting a sense of calm and Nordic elegance.

Design a minimalist area near the couch with an emphasis on open space and simplicity. Above the couch, hang a single, large-scale black and white photograph in a thin black frame, capturing the minimalist aesthetic. The surrounding space should remain uncluttered, with natural light as the primary decorative element, reflecting the minimalist ethos of functionality and simplicity.



## Description

InteriorDesign-Vue-Fast is a FastAPI-based application that utilizes Stable Diffusion, ControlNet, and Segment Anything Model (SAM) to generate and modify interior design images with AI.

This project allows users to:
- Generate interior design concepts based on text prompts.
- Apply segmentation and inpainting to modify furniture or decor.
- Use ControlNet to refine generated images with structured inputs.

## Features

- Stable Diffusion and ControlNet for generating detailed interior designs
- Segment Anything Model (SAM) for object segmentation in images
- Inpainting capabilities to modify and redesign furniture
- FastAPI backend for high-performance image processing
- Vue.js frontend integration for interactive design (optional)

## Installation

### Clone the Repository

git clone https://github.com/andresca94/InteriorDesign-Vue-Fast.git cd InteriorDesign-Vue-Fast


### Create a Conda Environment and Install Dependencies

conda env create -f environment.yml conda activate interior-design-env

## Running the API

Start the FastAPI server with:

uvicorn main:app --host 0.0.0.0 --port 9000

Once running, open `http://localhost:9000/docs` to test the API in an interactive UI.

## API Endpoints

| Endpoint                 | Method  | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `/generate-design/`      | `POST`  | Generate interior design from text.             |
| `/generate-mask/`        | `POST`  | Create object masks from bounding boxes.        |
| `/segment-with-box/`     | `POST`  | Segment objects in an image using SAM.          |
| `/inpaint-image/`        | `POST`  | Modify an object in an image using AI.          |

## Example Interior Design Prompts

Scandinavian Minimalism:
"Design a tranquil Scandinavian living space with a gallery wall of monochrome landscapes, oak armchairs, and woolen throws."

Victorian Elegance:
"Redesign the couch into a Victorian masterpiece with tufted velvet, mahogany legs, and intricate floral carvings."

Mid-Century Modern:
"Transform the living space into a sleek mid-century modern setting with teak wood furniture and warm earth tones."

## Security and API Keys

This project uses Hugging Face models. Ensure that your API keys are never hardcoded. Instead, store them in an `.env` file:

HUGGING_FACE_TOKEN=your_hugging_face_api_key

Then, load the key in Python:

import os hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

## Deployment

To deploy on a production server using Gunicorn:

gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

For Docker Deployment:

docker build -t interior-design-ai . docker run -p 9000:9000 interior-design-ai

## License

This project is licensed under the MIT License.

## Credits

- Stable Diffusion - RunwayML
- ControlNet - lllyasviel
- Segment Anything - Meta AI
