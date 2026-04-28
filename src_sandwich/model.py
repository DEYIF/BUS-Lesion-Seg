import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.ops import masks_to_boxes

# Import builds from existing modules
from src_unet.model import build_unet
from sam3.model_builder import build_sam3_image_model
# DistillModel import removed (unused, causes peft version conflict)

class SandwichModel(nn.Module):
    def __init__(self, unet1_ckpt, sam_ckpt, unet2_arch="resnet34",
                 unet2_weights="imagenet", sam_lora_ckpt=None):
        super().__init__()

        # === Stage 1: U-Net Teacher (Frozen) ===
        print("🥪 Layer 1: Loading U-Net Teacher (Frozen)...")
        self.unet1 = build_unet()
        # Load weights
        ckpt = torch.load(unet1_ckpt, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.unet1.load_state_dict(state_dict)
        self.unet1.eval()
        for p in self.unet1.parameters():
            p.requires_grad = False

        # === Stage 2: SAM 3 (Frozen, optionally LoRA-adapted) ===
        print(f"🥪 Layer 2: Loading SAM 3 (Frozen) from {sam_ckpt}...")
        self.sam = build_sam3_image_model(
            checkpoint_path=sam_ckpt,
            device='cpu',
            eval_mode=True, # Fully frozen
            enable_inst_interactivity=False
        )

        if sam_lora_ckpt is not None:
            from src_sandwich.model_v2 import _merge_lora_into_sam_decoder
            print(f"🥪 Layer 2: Merging LoRA weights from {sam_lora_ckpt} "
                  f"into SAM decoder (r=8, alpha=16, linear1+linear2)...")
            lora_ckpt = torch.load(sam_lora_ckpt, map_location='cpu', weights_only=False)
            lora_sd = lora_ckpt['model_state_dict'] if isinstance(lora_ckpt, dict) \
                      and 'model_state_dict' in lora_ckpt else lora_ckpt
            n_merged = _merge_lora_into_sam_decoder(self.sam, lora_sd, r=8, lora_alpha=16)
            print(f"🥪 Layer 2: merged {n_merged} LoRA adapter pairs.")

        # Freeze SAM completely
        for p in self.sam.parameters():
            p.requires_grad = False
            
        # SAM Preprocessing stats
        self.register_buffer("pixel_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        # === Stage 3: U-Net Student (Trainable) ===
        print(f"🥪 Layer 3: Building U-Net Student ({unet2_arch}, in_channels=4)...")
        self.unet2 = smp.Unet(
            encoder_name=unet2_arch,
            encoder_weights=unet2_weights,
            in_channels=4, # RGB + SAM Mask
            classes=1,
        )
        # This one is trainable by default.

    def get_box_from_mask(self, mask):
        """
        Generate Bounding Box (B, 4) from Mask (B, 1, H, W)
        """
        B, _, H, W = mask.shape
        boxes_list = []
        for i in range(B):
            m = mask[i, 0]
            if m.sum() == 0:
                 # Fallback: Center point or full image?
                 # Let's use full image for safety
                 box = torch.tensor([0, 0, W, H], device=mask.device, dtype=torch.float32)
            else:
                box = masks_to_boxes(m.unsqueeze(0))[0]
            boxes_list.append(box)
        return torch.stack(boxes_list)

    def forward_sam(self, images, boxes):
        """
        Run SAM3 inference (Adapted from DistillModel logic)
        images: (B, 3, H, W)
        boxes: (B, 4)
        """
        # 1. Resize to 1008x1008
        target_size = (1008, 1008)
        orig_H, orig_W = images.shape[2], images.shape[3]
        
        sam_input = (images - self.pixel_mean) / self.pixel_std
        sam_input = F.interpolate(sam_input, size=target_size, mode="bilinear", align_corners=False)
        
        # 2. Scale Boxes
        scale_y = target_size[0] / orig_H
        scale_x = target_size[1] / orig_W
        boxes_scaled = boxes.clone()
        boxes_scaled[:, 0] *= scale_x
        boxes_scaled[:, 1] *= scale_y
        boxes_scaled[:, 2] *= scale_x
        boxes_scaled[:, 3] *= scale_y
        
        # 3. Backbone
        backbone_out = self.sam.backbone.forward_image(sam_input)
        
        vis_feats = backbone_out["backbone_fpn"][-self.sam.num_feature_levels :]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.sam.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        
        # 4. Prompt Encoder
        if boxes_scaled.dim() == 2:
            boxes_scaled = boxes_scaled.unsqueeze(1) # (B, 1, 4)
            
        B = boxes_scaled.shape[0]
        N_boxes = boxes_scaled.shape[1]
        device = boxes_scaled.device
        
        from sam3.model.geometry_encoders import Prompt
        
        # Correct dimensions: (N, B, C)
        box_embeddings = boxes_scaled.permute(1, 0, 2) 
        box_labels = torch.ones((N_boxes, B), device=device, dtype=torch.long)
        box_mask = torch.zeros((B, N_boxes), device=device, dtype=torch.bool)
        
        geo_prompt = Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels
        )
        
        geo_feats, geo_masks = self.sam.geometry_encoder(
            geo_prompt=geo_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        # 5. Transformer Encoder
        prompt_pos_embed = torch.zeros_like(geo_feats)
        memory = self.sam.transformer.encoder(
            src=[x.clone() for x in img_feats],
            src_key_padding_mask=None,
            src_pos=[x.clone() for x in img_pos_embeds],
            prompt=geo_feats,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=geo_masks,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=None,
        )

        # 6. Transformer Decoder
        bs = memory["memory"].shape[1]
        query_embed = self.sam.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        hs, _, _, _ = self.sam.transformer.decoder(
            tgt=tgt,
            memory=memory["memory"],
            memory_key_padding_mask=memory["padding_mask"],
            pos=memory["pos_embed"],
            reference_boxes=None,
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=geo_feats,
            text_attention_mask=geo_masks,
            apply_dac=False, 
        )
        hs = hs.transpose(1, 2)
        
        # 7. Segmentation Head
        seg_head_outputs = self.sam.segmentation_head(
            backbone_feats=backbone_out["backbone_fpn"],
            obj_queries=hs,
            image_ids=torch.arange(bs, device=device),
            encoder_hidden_states=memory["memory"],
            act_ckpt_enable=False,
            prompt=geo_feats,
            prompt_mask=geo_masks,
        )
        
        # Take query 0
        sam_logits_low = seg_head_outputs["pred_masks"][:, 0:1, :, :]
        
        # Resize back to Original
        sam_logits = F.interpolate(
            sam_logits_low,
            size=(orig_H, orig_W),
            mode="bilinear",
            align_corners=False
        )
        return sam_logits

    def forward(self, images):
        """
        images: (B, 3, H, W)
        """
        if hasattr(images, "as_tensor"):
            images = images.as_tensor()

        # Step 1: U-Net 1 (Teacher)
        with torch.no_grad():
            unet1_logits = self.unet1(images)
            unet1_mask = (torch.sigmoid(unet1_logits) > 0.5).float()
            
            # Generate Box
            boxes = self.get_box_from_mask(unet1_mask) # (B, 4)
            """
            Gradient Truncation happens here implicitly because 
            argmax/thresholding/box-gen is not differentiable.
            """

        # Step 2: SAM 3 (Refiner 1)
        with torch.no_grad():
            sam_logits = self.forward_sam(images, boxes)
            sam_mask = torch.sigmoid(sam_logits) # (B, 1, H, W) Probabilities
            # Should we pass logits or probs? Probs is standard for concat.
            # Should we detach? Yes, SAM is frozen and box gen broke graph anyway.
            sam_mask = sam_mask.detach()

        # Step 3: Concatenation
        # Input (B, 3, H, W) + SAM Mask (B, 1, H, W) -> (B, 4, H, W)
        unet2_input = torch.cat([images, sam_mask], dim=1) 

        # Step 4: U-Net 2 (Student/Refiner 2)
        # This is the ONLY part where gradients flow
        unet2_logits = self.unet2(unet2_input) # (B, 1, H, W)
        
        return unet2_logits, sam_mask, unet1_mask
