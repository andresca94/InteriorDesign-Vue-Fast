<template>
  <div class="image-generator-container">
    <div class="upload-area">
      <input
        ref="fileInput"
        type="file"
        @change="onFileChange"
        hidden
      />
      <div v-if="!imagePreview && !generatedImage && !segmentedImage && !inpaintedImage" class="upload-instructions" @click="clickFileInput">
        Click here to select your images
      </div>
      <div ref="imageContainer" class="image-container" v-if="imagePreview || generatedImage || segmentedImage || inpaintedImage">
        <div v-if="isLoading" class="loading-overlay"><img src="@/assets/loading.gif" alt="Loading" class="loading-image"></div>
        <img
          :src="inpaintedImage ||segmentedImage || imagePreview || generatedImage"
          class="preview-image"
          alt="Preview Image"
          @load="onImageLoad"
          @mousedown="onMouseDown"
        />
        <div v-if="bboxCoords" class="box-overlay" :style="boxStyle"></div>
      </div>
    </div>
    <input
      type="text"
      v-model="concept"
      placeholder="Enter concept"
      class="concept-input"
    />
    <button @click="generateImage" class="generate-btn">
      Generate
    </button>
    <button v-if="bboxCoords" @click="inpaintImage" class="segment-btn">
      In-Paint Image
    </button>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      concept: '',
      imageId: null, // ID of the uploaded original image
      controlImage: null, // File object of the original image
      imagePreview: null,
      generatedImage: null,
      segmentedImage: null, // Blob URL of the segmented image
      bboxCoords: null,
      boxStart: null,
      drawingBox: false,
      boxStyle: {},
      clickCount: 0,
      segmentedImageId: null, // ID of the segmented image (if applicable)
      inpaintedImage: null,
      isLoading: false,
    };
  },
  methods: {
    clickFileInput() {
      this.$refs.fileInput.click();
    },
    onFileChange(e) {
      this.handleFiles(e.target.files);
    },
    handleFiles(files) {
  if (files.length > 0) {
    this.controlImage = files[0];
    this.previewImage(files[0]);
    this.uploadImage(); // Call the upload method immediately after setting the preview
  }
},

async uploadImage() {
  if (!this.controlImage) {
    console.error("No image selected for upload");
    return;
  }

  const formData = new FormData();
  formData.append("file", this.controlImage);

  try {
    const response = await axios.post('http://127.0.0.1:8000/upload-image/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    console.log("Image uploaded, server response:", response.data);
    this.imageId = response.data.image_id; // Assuming the server sends back an image_id
  } catch (error) {
    console.error("Error uploading image:", error.response?.data || error.message);
  }
},
previewImage(file) {
  const reader = new FileReader();
  reader.onload = e => {
    this.imagePreview = e.target.result;
    // Since FileReader is async, Vue may not detect changes automatically. Force update.
    this.$forceUpdate();
  };
  reader.onerror = e => {
    console.error('Error reading file:', e);
  };
  reader.readAsDataURL(file);
},
    onImageLoad() {
      console.log("Image is loaded and dimensions are set.");
    },
    onMouseDown(event) {
  this.clickCount += 1; // Increment the click count on each mouse down event

  if (this.clickCount === 1) {
    // First click: Start drawing the box
    this.boxStart = { x: event.offsetX, y: event.offsetY };
    this.drawingBox = true;
    this.$refs.imageContainer.addEventListener('mousemove', this.onMouseMove);
  } else if (this.clickCount === 2 && this.drawingBox) {
    // Second click: Finalize the box
    this.$refs.imageContainer.removeEventListener('mousemove', this.onMouseMove);
    this.drawingBox = false;
    this.bboxCoords = [this.boxStart, { x: event.offsetX, y: event.offsetY }];
    this.updateBoxOverlay();
  } else if (this.clickCount >= 3) {
    // Third click: Reset everything to start anew
    this.drawingBox = false;
    this.bboxCoords = null;
    this.boxStyle = {};
    this.clickCount = 0; // Reset the click count to allow the cycle to restart
  }
},
    onMouseMove(event) {
      if (this.drawingBox) {
        const currentPos = { x: event.offsetX, y: event.offsetY };
        this.bboxCoords = [this.boxStart, currentPos];
        this.updateBoxOverlay();
      }
    },
    updateBoxOverlay() {
      const minX = Math.min(this.bboxCoords[0].x, this.bboxCoords[1].x);
      const minY = Math.min(this.bboxCoords[0].y, this.bboxCoords[1].y);
      const width = Math.abs(this.bboxCoords[0].x - this.bboxCoords[1].x);
      const height = Math.abs(this.bboxCoords[0].y - this.bboxCoords[1].y);
      this.boxStyle = {
        top: `${minY}px`,
        left: `${minX}px`,
        width: `${width}px`,
        height: `${height}px`,
        position: 'absolute',
        border: '2px solid red',
        pointerEvents: 'none',
        zIndex: 10
      };
    },
    async generateImage() {
  this.isLoading = true;
  const formData = new FormData();
  formData.append('concept', this.concept);
  if (this.controlImage) {
    formData.append('control_image', this.controlImage, this.controlImage.name);
  }

  try {
    const response = await axios.post('http://127.0.0.1:9000/generate-design/', formData, {
      responseType: 'blob',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // Convert the blob response to a File object
    const generatedImageFile = new File([response.data], 'generatedImage.png', { type: 'image/png' });

    // Update controlImage to the generated image for upload
    this.controlImage = generatedImageFile;

    // Call uploadImage to upload the generated image
    await this.uploadImage();

    // Update UI to display the generated image
    this.inpaintedImage = null;
    this.generatedImage = URL.createObjectURL(generatedImageFile);
    this.imagePreview = null;
    this.segmentedImage = null;

  } catch (error) {
    console.error('Error generating image:', error);
  } finally {
    this.isLoading = false; // Stop loading when done
  }
},

async inpaintImage() {
  this.isLoading = true; 
  console.log("Starting the inpainting process...");

  if (!this.imageId || !this.bboxCoords || this.bboxCoords.length !== 2) {
    console.error('Image ID or bounding box coordinates are not properly set.');
    return;
  }

  console.log("Bounding box coordinates:", this.bboxCoords);

  const formattedBbox = {
    x: Math.min(this.bboxCoords[0].x, this.bboxCoords[1].x),
    y: Math.min(this.bboxCoords[0].y, this.bboxCoords[1].y),
    width: Math.abs(this.bboxCoords[0].x - this.bboxCoords[1].x),
    height: Math.abs(this.bboxCoords[0].y - this.bboxCoords[1].y)
  };

  try {
    const segmentResponse = await axios.post('http://127.0.0.1:8000/segment_with_box/', {
      image_id: this.imageId,
      box: formattedBbox
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (segmentResponse.data.status === "Success") {
      console.log("Segmentation successful. Proceeding to inpainting...");

      const segmentedImageResponse = await axios.get(`http://127.0.0.1:8000/image/get/${segmentResponse.data.output_img}`, { responseType: 'blob' });
      const maskBlob = segmentedImageResponse.data;

      const maskFile = new File([maskBlob], 'mask.png', {type: 'image/png'});

      const originalImageResponse = await axios.get(`http://127.0.0.1:8000/image/get/${this.imageId}`, { responseType: 'blob' });
      const originalImageBlob = originalImageResponse.data;
      const originalImageFile = new File([originalImageBlob], 'original.png', {type: 'image/png'});

      const formData = new FormData();
      formData.append('prompt', this.concept);
      formData.append('original_image', originalImageFile);
      formData.append('mask_image', maskFile);

      const inpaintResponse = await axios.post('http://127.0.0.1:9000/inpaint-image/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      // Convert the blob response to a File object for the inpainted image
      const inpaintedImageFile = new File([inpaintResponse.data], 'inpaintedImage.png', { type: 'image/png' });

      // Update controlImage with the new inpainted image
      this.controlImage = inpaintedImageFile;

      // Update the UI to display the inpainted image
      this.inpaintedImage = URL.createObjectURL(inpaintResponse.data);

      // Clear other image previews or states if necessary
      this.$nextTick(() => {
        this.generatedImage = null;
        this.segmentedImage = null;
        this.imagePreview = null;
      });
    } else {
      console.error('Segmentation was not successful:', segmentResponse.data);
    }
  } catch (error) {
    console.error('Error in inpainting process:', error.response?.data || error.message);
  }  finally {
    this.isLoading = false; // Stop loading when done
    // Before setting isLoading to false, clear the bounding box coordinates and styles
    this.bboxCoords = null; // Clear the bounding box coordinates
    this.boxStyle = {}; // Reset the box style to its initial state
  }
},

    async segmentImage() {
  console.log("Starting the segmentation process...");

  if (!this.imageId || !this.bboxCoords || this.bboxCoords.length !== 2) {
    console.error('Image ID or bounding box coordinates are not properly set.');
    return;
  }

  console.log("Bounding box coordinates:", this.bboxCoords);

  const formattedBbox = {
    x: Math.min(this.bboxCoords[0].x, this.bboxCoords[1].x),
    y: Math.min(this.bboxCoords[0].y, this.bboxCoords[1].y),
    width: Math.abs(this.bboxCoords[0].x - this.bboxCoords[1].x),
    height: Math.abs(this.bboxCoords[0].y - this.bboxCoords[1].y)
  };

  const payload = {
    image_id: this.imageId,
    box: formattedBbox
  };

  console.log("Sending segmentation request to backend with payload:", payload);

  try {
    const response = await axios.post('http://127.0.0.1:8000/segment_with_box/', payload, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    console.log("Segmentation response received from backend:", response.data);

    if (response.data.status === "Success") {
      // Fetch the image from the backend using the returned image ID
      this.fetchSegmentedImage(response.data.output_img);
      // After initiating segmentation, hide the red box by setting bboxCoords to null
      this.bboxCoords = null;
      this.boxStyle = {}; // Also reset the box style to clear it from the view
    } else {
      console.error('Segmentation was not successful:', response.data);
    }
  } catch (error) {
    console.error('Error in segmenting image:', error.response?.data || error.message);
  }
},

async fetchSegmentedImage(imageId) {
      try {
        const response = await axios.get(`http://127.0.0.1:8000/image/get/${imageId}`, {
          responseType: 'blob',
        });
        this.segmentedImage = URL.createObjectURL(new Blob([response.data], { type: 'image/png' }));
        this.segmentedImageId = imageId; // Store the segmented image ID if needed
      } catch (error) {
        console.error('Error fetching segmented image:', error.response?.data || error.message);
      }
    },

formatBboxForBackend(bboxCoords) {
  // Assuming bboxCoords is an array of two points [ [x1, y1], [x2, y2] ]
  const [start, end] = bboxCoords;
  // Create the bounding box in the format expected by the backend: [x1, y1, x2, y2]
  return [start.x, start.y, end.x, end.y];
},
  },
  mounted() {
    if (this.$refs.imageContainer) {
      this.$refs.imageContainer.addEventListener('mouseup', this.onMouseUp);
    }
  },
  unmounted() {
    if (this.$refs.imageContainer) {
      this.$refs.imageContainer.removeEventListener('mouseup', this.onMouseUp);
    }
  }
};
</script>



<style scoped>
.image-generator-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
  gap: 20px;
}

.upload-area {
  border: 2px dashed #ccc;
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  cursor: pointer;
  position: relative;
  transition: background-color 0.2s;
}

.concept-input {
  width: 80%;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
}

.generate-btn, .segment-btn {
  padding: 10px 20px;
  border: none;
  background-color: #4CAF50;
  color: white;
  border-radius: 5px;
  cursor: pointer;
}

.segment-btn {
  background-color: #008CBA;
}

.image-container {
  position: relative;
}

.box-overlay {
  position: absolute;
  border: 2px solid red;
  pointer-events: none;
  z-index: 10;
}

.preview-image {
  max-width: 100%;
  max-height: 500px;
  border: 1px solid #ccc;
  border-radius: 5px;
  object-fit: contain;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(255, 255, 255, 0.7); /* Semi-transparent white */
  z-index: 20; /* Ensure it covers other elements */
}

.loading-image {
  width: 50px; /* Adjust based on your GIF size */
  height: 50px; /* Adjust based on your GIF size */
}
</style>




  