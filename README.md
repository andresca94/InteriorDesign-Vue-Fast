# InteriorDesign-Vue-Fast

ControlNet Stable Diffusion for Interior Design

Design a tranquil Scandinavian living space where simplicity and comfort converge. Picture a soft woolen herringbone throw casually draped over a nearby oak armchair, with a series of potted succulents in clean-lined concrete planters arranged on the floor beside the couch. Above the couch, imagine a gallery wall of monochromatic landscapes in slim black frames, promoting a sense of calm and Nordic elegance.


Design a minimalist area near the couch with an emphasis on open space and simplicity. Above the couch, hang a single, large-scale black and white photograph in a thin black frame, capturing the minimalist aesthetic. The surrounding space should remain uncluttered, with natural light as the primary decorative element, reflecting the minimalist ethos of functionality and simplicity.


Segment Anything Model + In-painting Stable Diffusion



Reshape the Couch to capture the essence of Victorian elegance. In-paint the fabric to be a rich, damask in a burgundy hue with intricate floral patterns woven throughout. The Couch's form should be modified to include a curved backrest and scrolled armrests, with button-tufting to create a diamond pattern. Legs should be transformed to resemble polished mahogany with classic claw-and-ball feet, characteristic of 19th-century craftsmanship.

Revamp the Couch to embody the sleek minimalism of Mid-Century Modern design. In-paint the upholstery to be a smooth, premium leather in a muted olive green, reflecting the era's affinity for organic hues. The Couch's silhouette should be streamlined, with a low-profile and clean lines, integrating tapered legs made of teak wood. Cushions should be firm yet comfortable, showcasing subtle button tufting for a touch of sophistication. The overall design must echo the functional elegance and timeless appeal of the 1950s and 1960s furniture trends

# InteriorDesign-Vue-Fast

ControlNet Stable Diffusion for Interior Design

Design a tranquil Scandinavian living space where simplicity and comfort converge. Picture a soft woolen herringbone throw casually draped over a nearby oak armchair, with a series of potted succulents in clean-lined concrete planters arranged on the floor beside the couch. Above the couch, imagine a gallery wall of monochromatic landscapes in slim black frames, promoting a sense of calm and Nordic elegance.


Design a minimalist area near the couch with an emphasis on open space and simplicity. Above the couch, hang a single, large-scale black and white photograph in a thin black frame, capturing the minimalist aesthetic. The surrounding space should remain uncluttered, with natural light as the primary decorative element, reflecting the minimalist ethos of functionality and simplicity.


markdown
Copy
Edit
# InteriorDesign-Vue-Fast

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

shell
Copy
Edit

### Create a Conda Environment and Install Dependencies

conda env create -f environment.yml conda activate interior-design-env

sql
Copy
Edit

## Running the API

Start the FastAPI server with:

uvicorn main:app --host 0.0.0.0 --port 9000

perl
Copy
Edit

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

lua
Copy
Edit

Then, load the key in Python:

import os hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

csharp
Copy
Edit

## Deployment

To deploy on a production server using Gunicorn:

gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

yaml
Copy
Edit

For Docker Deployment:

docker build -t interior-design-ai . docker run -p 9000:9000 interior-design-ai

markdown
Copy
Edit

## License

This project is licensed under the MIT License.

## Credits

- Stable Diffusion - RunwayML
- ControlNet - lllyasviel
- Segment Anything - Meta AI

## Contact

For feedback or support, open an issue in the repository.


Segment Anything Model + In-painting Stable Diffusion


Reshape the Couch to capture the essence of Victorian elegance. In-paint the fabric to be a rich, damask in a burgundy hue with intricate floral patterns woven throughout. The Couch's form should be modified to include a curved backrest and scrolled armrests, with button-tufting to create a diamond pattern. Legs should be transformed to resemble polished mahogany with classic claw-and-ball feet, characteristic of 19th-century craftsmanship.

Revamp the Couch to embody the sleek minimalism of Mid-Century Modern design. In-paint the upholstery to be a smooth, premium leather in a muted olive green, reflecting the era's affinity for organic hues. The Couch's silhouette should be streamlined, with a low-profile and clean lines, integrating tapered legs made of teak wood. Cushions should be firm yet comfortable, showcasing subtle button tufting for a touch of sophistication. The overall design must echo the functional elegance and timeless appeal of the 1950s and 1960s furniture trends





