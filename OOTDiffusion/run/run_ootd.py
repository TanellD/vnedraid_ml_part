from flask import Flask, request, jsonify, send_file
from pathlib import Path
import sys
import os
import base64
from PIL import Image
import io
import json
import traceback
import torch
import gc

# Add project root to path
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC
from utils_ootd import get_mask_location

app = Flask(__name__)

class OOTDServer:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.models = {}
        self.openpose_model = None
        self.parsing_model = None
        self.category_dict = ['upperbody', 'lowerbody', 'dress']
        self.category_dict_utils = ['upper_body', 'lower_body', 'dresses']
        
        # Create output directory
        os.makedirs('./images_output', exist_ok=True)
        
        print("🚀 Initializing OOTD Server...")
        self._load_preprocessing_models()
        
    def _load_preprocessing_models(self):
        """Load OpenPose and Parsing models"""
        print("📦 Loading OpenPose model...")
        self.openpose_model = OpenPose(self.gpu_id)
        
        print("📦 Loading Parsing model...")
        self.parsing_model = Parsing(self.gpu_id)
        
        print("✅ Preprocessing models loaded!")
        
    def _load_ootd_model(self, model_type):
        """Load OOTD model if not already loaded"""
        if model_type not in self.models:
            print(f"📦 Loading OOTD {model_type.upper()} model...")
            
            # Clear memory before loading
            torch.cuda.empty_cache()
            gc.collect()
            
            if model_type == "hd":
                self.models[model_type] = OOTDiffusionHD(self.gpu_id)
            elif model_type == "dc":
                self.models[model_type] = OOTDiffusionDC(self.gpu_id)
            else:
                raise ValueError("model_type must be 'hd' or 'dc'!")
                
            print(f"✅ OOTD {model_type.upper()} model loaded!")
        
        return self.models[model_type]
    
    def _decode_base64_image(self, base64_string):
        """Decode base64 string to PIL Image"""
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    
    def _encode_image_to_base64(self, image):
        """Encode PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64
    
    def _save_image_with_name(self, image, filename):
        """Save image and return the path"""
        filepath = f"./images_output/{filename}"
        image.save(filepath)
        return filepath
    # Also fix the process_ootd method in your OOTDServer class:
    def process_ootd(self, 
                    model_image, 
                    cloth_image, 
                    model_type="dc", 
                    category=0, 
                    scale=2.0, 
                    steps=20, 
                    samples=1, 
                    seed=-1):
        """Process OOTD inference"""
        
        try:
            # Validate inputs
            if model_type == 'hd' and category != 0:
                raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")
            
            # Load OOTD model
            model = self._load_ootd_model(model_type)
            
            # Resize images
            cloth_img = cloth_image.resize((768, 1024))
            model_img = model_image.resize((768, 1024))
            
            # Generate keypoints and parsing
            print("🔍 Generating keypoints...")
            keypoints = self.openpose_model(model_img.resize((384, 512)))
            
            print("🎭 Generating parsing mask...")
            model_parse, *_ = self.parsing_model(model_img.resize((384, 512)))
            
            # Get mask
            mask, mask_gray = get_mask_location(
                model_type, 
                self.category_dict_utils[category], 
                model_parse, 
                keypoints
            )
            mask = mask.resize((768, 1024), Image.NEAREST)
            mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
            
            # Create masked image
            masked_vton_img = Image.composite(mask_gray, model_img, mask)
            
            # Run OOTD inference
            print("🎨 Running OOTD inference...")
            images = model(
                model_type=model_type,
                category=self.category_dict[category],
                image_garm=cloth_img,
                image_vton=masked_vton_img,
                mask=mask,
                image_ori=model_img,
                num_samples=samples,
                num_steps=steps,
                image_scale=scale,
                seed=seed,
            )
            
            # Save mask for debugging
            mask_path = self._save_image_with_name(masked_vton_img, f"mask_{model_type}.jpg")
            
            # Save generated images and prepare response
            result_images = []
            for idx, generated_image in enumerate(images):  # Fixed: use 'generated_image' instead of 'image'
                filename = f"out_{model_type}_{idx}.png"
                filepath = self._save_image_with_name(generated_image, filename)
                
                # Convert to base64 for API response
                image_base64 = self._encode_image_to_base64(generated_image)
                result_images.append({
                    "index": idx,
                    "filename": filename,
                    "filepath": filepath,
                    "image_base64": image_base64
                })
            
            return {
                "success": True,
                "model_type": model_type,
                "category": self.category_dict[category],
                "num_generated": len(images),
                "mask_path": mask_path,
                "images": result_images
            }
            
        except Exception as e:
            print(f"❌ Error in OOTD processing: {str(e)}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    # def process_ootd(self, 
    #                 model_image, 
    #                 cloth_image, 
    #                 model_type="dc", 
    #                 category=0, 
    #                 scale=2.0, 
    #                 steps=20, 
    #                 samples=1, 
    #                 seed=-1):
    #     """Process OOTD inference"""
        
    #     try:
    #         # Validate inputs
    #         if model_type == 'hd' and category != 0:
    #             raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")
            
    #         # Load OOTD model
    #         model = self._load_ootd_model(model_type)
            
    #         # Resize images
    #         cloth_img = cloth_image.resize((768, 1024))
    #         model_img = model_image.resize((768, 1024))
            
    #         # Generate keypoints and parsing
    #         print("🔍 Generating keypoints...")
    #         keypoints = self.openpose_model(model_img.resize((384, 512)))
            
    #         print("🎭 Generating parsing mask...")
    #         model_parse, *_ = self.parsing_model(model_img.resize((384, 512)))
            
    #         # Get mask
    #         mask, mask_gray = get_mask_location(
    #             model_type, 
    #             self.category_dict_utils[category], 
    #             model_parse, 
    #             keypoints
    #         )
    #         mask = mask.resize((768, 1024), Image.NEAREST)
    #         mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
            
    #         # Create masked image
    #         masked_vton_img = Image.composite(mask_gray, model_img, mask)
            
    #         # Run OOTD inference
    #         print("🎨 Running OOTD inference...")
    #         images = model(
    #             model_type=model_type,
    #             category=self.category_dict[category],
    #             image_garm=cloth_img,
    #             image_vton=masked_vton_img,
    #             mask=mask,
    #             image_ori=model_img,
    #             num_samples=samples,
    #             num_steps=steps,
    #             image_scale=scale,
    #             seed=seed,
    #         )
            
    #         # Save mask for debugging
    #         mask_path = self._save_image_with_name(masked_vton_img, f"mask_{model_type}.jpg")
            
    #         # Save generated images and prepare response
    #         result_images = []
    #         filename = f"out_{model_type}_0.png"
    #         filepath = self._save_image_with_name(image, filename)
            
    #         # Convert to base64 for API response
    #         image_base64 = self._encode_image_to_base64(image)
    #         result_images = {
    #             "filename": filename,
    #             "image_base64": image_base64
    #         }
            
    #         return {
    #             "success": True,
    #             "model_type": model_type,
    #             "category": self.category_dict[category],
    #             "mask_path": mask_path,
    #             "image": result_images
    #         }
            
    #     except Exception as e:
    #         print(f"❌ Error in OOTD processing: {str(e)}")
    #         traceback.print_exc()
    #         return {
    #             "success": False,
    #             "error": str(e),
    #             "traceback": traceback.format_exc()
    #         }

# Initialize server
ootd_server = OOTDServer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": list(ootd_server.models.keys()),
        "gpu_id": ootd_server.gpu_id
    })

@app.route('/generate', methods=['POST'])
def generate_ootd():
    """Main OOTD generation endpoint - returns first image only"""
    try:
        data = request.get_json()
        
        # Extract parameters
        model_image_b64 = data.get('model_image')  # base64 encoded
        cloth_image_b64 = data.get('cloth_image')  # base64 encoded
        model_type = data.get('model_type', 'dc')
        category = data.get('category', 0)
        scale = data.get('scale', 2.0)
        steps = data.get('steps', 20)
        samples = data.get('samples', 1)
        seed = data.get('seed', 42)
        
        # Validate required inputs
        if not model_image_b64 or not cloth_image_b64:
            return jsonify({
                "success": False,
                "error": "model_image and cloth_image are required"
            }), 400
        
        # Decode images
        model_image = ootd_server._decode_base64_image(model_image_b64)
        cloth_image = ootd_server._decode_base64_image(cloth_image_b64)
        
        # Process OOTD - force samples=1 to get only first image
        result = ootd_server.process_ootd(
            model_image=model_image,
            cloth_image=cloth_image,
            model_type=model_type,
            category=category,
            scale=scale,
            steps=steps,
            samples=1,  # Force only 1 sample
            seed=seed
        )
        
        # Return only the first image info
        if result["success"] and result["images"]:
            first_image = result["images"][0]
            return jsonify({
                "success": True,
                "model_type": result["model_type"],
                "category": result["category"],
                "image": {
                    "filename": first_image["filename"],
                    "filepath": first_image["filepath"],
                    "image_base64": first_image["image_base64"]
                },
                "mask_path": result["mask_path"]
            })
        else:
            return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/generate_from_paths', methods=['POST'])
def generate_from_paths():
    """Generate OOTD from file paths (like your original script)"""
    try:
        data = request.get_json()
        
        # Extract parameters
        model_path = data.get('model_path')
        cloth_path = data.get('cloth_path')
        model_type = data.get('model_type', 'dc')
        category = data.get('category', 0)
        scale = data.get('scale', 2.0)
        steps = data.get('steps', 20)
        samples = data.get('samples', 1)
        seed = data.get('seed', -1)
        
        # Validate required inputs
        if not model_path or not cloth_path:
            return jsonify({
                "success": False,
                "error": "model_path and cloth_path are required"
            }), 400
        
        # Load images
        model_image = Image.open(model_path)
        cloth_image = Image.open(cloth_path)
        
        # Process OOTD
        result = ootd_server.process_ootd(
            model_image=model_image,
            cloth_image=cloth_image,
            model_type=model_type,
            category=category,
            scale=scale,
            steps=steps,
            samples=samples,
            seed=seed
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_image(filename):
    """Download generated images"""
    try:
        filepath = f"./images_output/{filename}"
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Preload a specific model type"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'dc')
        
        model = ootd_server._load_ootd_model(model_type)
        
        return jsonify({
            "success": True,
            "message": f"Model {model_type} loaded successfully",
            "models_loaded": list(ootd_server.models.keys())
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("🌟 Starting OOTD Server...")
    print("📍 Endpoints:")
    print("   GET  /health - Health check")
    print("   POST /generate - Generate from base64 images")
    print("   POST /generate_from_paths - Generate from file paths")
    print("   POST /load_model - Preload model")
    print("   GET  /download/<filename> - Download generated images")
    print()
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
