import torch
import torch.nn.functional as F
from torch import nn
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path
import importlib.util
import torchgeometry as tgm
import sys

class VITONService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # VITON-HD path
        current_dir = Path(__file__).parent.parent
        viton_path = (current_dir / 'VITON-HD').resolve()
        
        # Import networks
        networks_path = viton_path / 'networks.py'
        if not networks_path.exists():
            raise FileNotFoundError(f"Networks module not found at {networks_path}")
            
        spec = importlib.util.spec_from_file_location("networks", networks_path)
        networks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(networks)
        
        # Import utils
        utils_path = viton_path / 'utils.py'
        if not utils_path.exists():
            raise FileNotFoundError(f"Utils module not found at {utils_path}")
            
        spec = importlib.util.spec_from_file_location("utils", utils_path)
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        
        # Model classes
        self.SegGenerator = networks.SegGenerator
        self.GMM = networks.GMM
        self.ALIASGenerator = networks.ALIASGenerator
        self.gen_noise = utils.gen_noise
        
        # Load models with opt matching test.py
        self.opt = self._get_opt()
        self.seg_model = self._load_seg_model()
        self.gmm_model = self._load_gmm_model()
        self.alias_model = self._load_alias_model()
        self._models_loaded = True
        
    def _get_opt(self):
        # Mimic test.py's get_opt()
        class Opt:
            pass
        opt = Opt()
        opt.load_height = 1024
        opt.load_width = 768
        opt.semantic_nc = 13
        opt.init_type = 'xavier'
        opt.init_variance = 0.02
        opt.grid_size = 5
        opt.norm_G = 'spectralaliasinstance'
        opt.ngf = 64
        opt.num_upsampling_layers = 'most'
        return opt
    
    def _load_seg_model(self):
        model = self.SegGenerator(self.opt, input_nc=self.opt.semantic_nc + 8, output_nc=self.opt.semantic_nc)
        checkpoint_path = Path('./VITON-HD/checkpoints/seg_final.pth').resolve()
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))
        model.to(self.device).eval()
        return model
    
    def _load_gmm_model(self):
        model = self.GMM(self.opt, inputA_nc=7, inputB_nc=3)
        checkpoint_path = Path('./VITON-HD/checkpoints/gmm_final.pth').resolve()
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))
        model.to(self.device).eval()
        return model
    
    def _load_alias_model(self):
        self.opt.semantic_nc = 7  # Adjusted for ALIAS as in test.py
        model = self.ALIASGenerator(self.opt, input_nc=9)
        checkpoint_path = Path('./VITON-HD/checkpoints/alias_final.pth').resolve()
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))
        model.to(self.device).eval()
        self.opt.semantic_nc = 13  # Reset for consistency
        return model
    
    def _generate_cloth_mask(self, cloth_img):
        """Generate better cloth mask using OpenCV processing"""
        # Convert to numpy for OpenCV processing
        cloth_np = np.array(cloth_img)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cloth_np, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to tensor
        mask_tensor = torch.from_numpy(binary).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        mask_tensor = F.interpolate(mask_tensor, size=(1024, 768), mode='bilinear')
        mask_tensor = mask_tensor.to(self.device)
        
        return mask_tensor
    
    def _generate_pose_map(self, person_img):
        """
        Generate a basic pose map as a placeholder
        In a real implementation, this would use a proper pose estimator
        """
        # Here we'll create a simple placeholder that's better than all zeros
        pose_map = np.zeros((1024, 768, 3), dtype=np.uint8)
        
        # Convert person image to numpy
        person_np = np.array(person_img.resize((768, 1024)))
        
        # Simple body keypoints (this is a placeholder - a real system would detect these)
        # Format: [(x, y, confidence), ...]
        keypoints = [
            (384, 200, 1.0),  # head
            (384, 300, 1.0),  # neck
            (384, 400, 1.0),  # chest
            (300, 400, 1.0),  # left shoulder
            (468, 400, 1.0),  # right shoulder
            (250, 500, 1.0),  # left elbow
            (518, 500, 1.0),  # right elbow
            (200, 600, 1.0),  # left wrist
            (568, 600, 1.0),  # right wrist
            (384, 600, 1.0),  # waist
            (350, 750, 1.0),  # left knee
            (418, 750, 1.0),  # right knee
            (350, 900, 1.0),  # left ankle
            (418, 900, 1.0)   # right ankle
        ]
        
        # Draw keypoints and connections
        for kp in keypoints:
            x, y, conf = kp
            cv2.circle(pose_map, (x, y), 6, (0, 255, 255), -1)
        
        # Draw some connections to mimic a skeleton
        connections = [
            (0, 1), (1, 2), (2, 9),  # Head to waist
            (2, 3), (3, 5), (5, 7),  # Left arm
            (2, 4), (4, 6), (6, 8),  # Right arm
            (9, 10), (10, 12),       # Left leg
            (9, 11), (11, 13)        # Right leg
        ]
        
        for conn in connections:
            pt1 = (keypoints[conn[0]][0], keypoints[conn[0]][1])
            pt2 = (keypoints[conn[1]][0], keypoints[conn[1]][1])
            cv2.line(pose_map, pt1, pt2, (0, 255, 0), 3)
        
        # Convert to tensor
        pose_tensor = torch.from_numpy(pose_map.transpose(2, 0, 1)).float() / 255.0
        pose_tensor = pose_tensor.unsqueeze(0).to(self.device)
        
        return pose_tensor
    
    def _generate_parse_map(self, person_img):
        """
        Generate a basic parse map as a placeholder
        In a real implementation, this would use a proper human parser
        """
        # Convert person image to numpy and resize
        person_np = np.array(person_img.resize((768, 1024)))
        
        # Create an empty parse map
        parse_map = np.zeros((1024, 768), dtype=np.uint8)
        
        # Simple segmentation based on color and position
        # This is a very basic approach - a real system would use a trained segmentation model
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(person_np, cv2.COLOR_RGB2HSV)
        
        # Basic regions by position (this is an extremely simplified approach)
        # Background
        parse_map[:, :] = 0
        
        # Face/hair region (upper part)
        parse_map[100:300, 284:484] = 1  # Hair
        parse_map[200:350, 334:434] = 2  # Face
        
        # Torso region (middle part)
        parse_map[350:650, 284:484] = 3  # Upper clothes
        
        # Arms
        parse_map[350:600, 184:284] = 5  # Left arm
        parse_map[350:600, 484:584] = 6  # Right arm
        
        # Lower body
        parse_map[650:900, 284:484] = 4  # Pants
        
        # Convert to one-hot encoding
        parse_tensor = torch.zeros(13, 1024, 768, dtype=torch.float, device=self.device)
        for i in range(13):
            parse_tensor[i] = (torch.from_numpy(parse_map) == i).float()
        
        parse_tensor = parse_tensor.unsqueeze(0)  # Add batch dimension
        
        return parse_tensor
    
    def _generate_agnostic_image(self, img_tensor, parse_tensor):
        """
        Generate an image without the clothing (agnostic image)
        """
        # Create a mask for the clothing region (upper clothes, index 3)
        clothing_mask = parse_tensor[:, 3:4]
        
        # Create agnostic image by masking out the clothing region
        agnostic = img_tensor.clone()
        agnostic = agnostic * (1 - clothing_mask)
        
        # Could also fill with a neutral color where clothing was
        # This is simplified - a real implementation would be more sophisticated
        
        return agnostic
    
    def _generate_parse_agnostic(self, parse_tensor):
        """
        Generate a parse map without the clothing
        """
        parse_agnostic = parse_tensor.clone()
        
        # Remove upper clothes (usually index 3)
        parse_agnostic[:, 3] = 0
        
        return parse_agnostic
    
    def process_images(self, person_img, cloth_img):
        if not self._models_loaded:
            raise RuntimeError("Models not properly loaded")
            
        debug_dir = Path("./debug").resolve()
        os.makedirs(debug_dir, exist_ok=True)
        
        # Preprocess inputs with improved methods
        c = self.transform(cloth_img).unsqueeze(0).to(self.device)
        cm = self._generate_cloth_mask(cloth_img)
        
        # Process person image to get all required inputs
        img = self.transform(person_img).unsqueeze(0).to(self.device)
        parse = self._generate_parse_map(person_img)
        pose = self._generate_pose_map(person_img)
        img_agnostic = self._generate_agnostic_image(img, parse)
        parse_agnostic = self._generate_parse_agnostic(parse)
        
        # Save initial inputs for debugging
        self._save_tensor_as_image(c, debug_dir / "input_cloth.png")
        self._save_tensor_as_image(cm, debug_dir / "cloth_mask.png")
        self._save_tensor_as_image(img_agnostic, debug_dir / "img_agnostic.png")
        self._save_tensor_as_image(pose, debug_dir / "pose.png")
        self._save_tensor_as_image(parse.sum(dim=1, keepdim=True), debug_dir / "parse.png")
        
        with torch.no_grad():
            # Part 1: Segmentation generation (matching test.py)
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, self.gen_noise(cm_down.size())), dim=1)
            
            parse_pred_down = self.seg_model(seg_input)
            up = nn.Upsample(size=(self.opt.load_height, self.opt.load_width), mode='bilinear')
            gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(self.device)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]
            
            parse_old = torch.zeros(1, 13, self.opt.load_height, self.opt.load_width, dtype=torch.float, device=self.device)
            parse_old.scatter_(1, parse_pred, 1.0)
            
            parse = torch.zeros(1, 7, self.opt.load_height, self.opt.load_width, dtype=torch.float, device=self.device)
            labels = {
                0: [0],  # Background
                1: [2, 4, 7, 8, 9, 10, 11],  # Paste
                2: [3],  # Upper
                3: [1],  # Hair
                4: [5],  # Left arm
                5: [6],  # Right arm
                6: [12]  # Noise
            }
            for j in range(len(labels)):
                for label in labels[j]:
                    parse[:, j] += parse_old[:, label]
            
            self._save_tensor_as_image(parse.sum(dim=1, keepdim=True), debug_dir / "generated_parse.png")
            
            # Part 2: Clothes deformation (GMM) - exactly as in test.py
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
            
            _, warped_grid = self.gmm_model(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')
            
            self._save_tensor_as_image(warped_c, debug_dir / "warped_c.png")
            self._save_tensor_as_image(warped_cm, debug_dir / "warped_cm.png")
            
            # Part 3: Try-on synthesis (ALIAS) - exactly as in test.py
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask
            
            alias_input = torch.cat((img_agnostic, pose, warped_c), dim=1)
            output = self.alias_model(alias_input, parse, parse_div, misalign_mask)
            
            self._save_tensor_as_image(output, debug_dir / "raw_output.png")
        
        # Post-process
        output = output.squeeze().cpu().numpy()
        output = (output + 1) / 2
        output = (output * 255).astype(np.uint8)
        output = output.transpose(1, 2, 0)
        Image.fromarray(output).save(debug_dir / "final_processed_output.png")
        
        return Image.fromarray(output)
    
    def _save_tensor_as_image(self, tensor, path):
        if len(tensor.shape) == 4:
            if tensor.shape[1] == 3:
                img = tensor[0].cpu().detach().permute(1, 2, 0)
                img = ((img + 1) / 2 * 255).numpy().astype(np.uint8)
                Image.fromarray(img).save(path)
            else:
                if tensor.shape[1] >= 3:
                    img = tensor[0, :3].cpu().detach().permute(1, 2, 0)
                    img = ((img + 1) / 2 * 255).numpy().astype(np.uint8)
                    Image.fromarray(img).save(path)
                else:
                    img = tensor[0, 0].cpu().detach()
                    img = ((img + 1) / 2 * 255).numpy().astype(np.uint8)
                    Image.fromarray(img).save(path)
    
    def is_ready(self):
        return self._models_loaded