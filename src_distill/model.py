import torch
import torch.nn as nn
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from sam3.model_builder import build_sam3_image_model
from sam3.model.geometry_encoders import Prompt
import os
import sys

# 引用你的 U-Net
from src_unet.model import build_unet

class DistillModel(nn.Module):
    def __init__(self, unet_ckpt, sam_ckpt, sam_config=None):
        """
        sam_ckpt: path to sam3.pt
        sam_config: Ignored for SAM3 (config is implicit/default for now)
        """
        super().__init__()

        # === 1. Teacher: U-Net (冻结) ===
        print("🧊 Loading Teacher (U-Net)...")
        self.teacher = build_unet()
        # 加载权重 (处理可能的 key 不匹配)
        ckpt = torch.load(unet_ckpt, map_location='cpu')
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.teacher.load_state_dict(state_dict)
        
        # 彻底冻结 Teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # === 2. Student: SAM 3 (加载 sam3.pt) ===
        print(f"🔥 Loading Student (SAM 3 from {sam_ckpt})...")
        
        # 使用 SAM 3 的构建函数
        self.student = build_sam3_image_model(
            checkpoint_path=sam_ckpt,
            device='cpu', # 先加载到 CPU
            eval_mode=False, # We want to train parts of it
            enable_inst_interactivity=False # We only use core segmentation
        )
        
        # Default pixel mean/std from SAM3 (0.5, 0.5, 0.5)
        self.register_buffer("pixel_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        # === 3. 冻结策略 (显存优化) ===
        # SAM 的 Image Encoder 非常大，建议冻结，只训练 Mask Decoder
        # 这样显存占用小，且不容易破坏 SAM 原有的预训练特征
        
        # Freezing Backbone
        for p in self.student.backbone.parameters():
            p.requires_grad = False
        # Freezing Geometry Encoder (Prompt Encoder) ?? Usually yes.
        for p in self.student.geometry_encoder.parameters():
            p.requires_grad = False
        # Freezing Transformer Encoder ?? Usually yes for SAM fine-tuning.
        for p in self.student.transformer.encoder.parameters():
            p.requires_grad = False
            
        # 仅让 Mask Decoder 可训练 (Transformer Decoder + Segmentation Head)
        for p in self.student.transformer.decoder.parameters():
            p.requires_grad = True
        if self.student.segmentation_head is not None:
             for p in self.student.segmentation_head.parameters():
                 p.requires_grad = True
        if self.student.dot_prod_scoring is not None:
             for p in self.student.dot_prod_scoring.parameters():
                 p.requires_grad = True

    def forward(self, images, boxes=None):
        """
        images: (B, 3, H, W) - Tensor. Assumed 0-1 range (float).
        boxes: (B, 4) - Box coords.
        """
        if hasattr(images, "as_tensor"):
            images = images.as_tensor()
            
        # 1. Teacher 推理
        with torch.no_grad():
            teacher_logits = self.teacher(images) # (B, 1, H, W)
            
        if boxes is None:
             raise ValueError("Boxes must be provided! Use GT boxes or Teacher-predicted boxes.")

        # 2. Student (SAM) 推理
        # SAM3 Preprocessing: (x - 0.5) / 0.5
        # images are assumed 0-1.
        sam_input = (images - self.pixel_mean) / self.pixel_std

        # Resize to 1008x1008 (SAM3 default resolution)
        target_sam_size = (1008, 1008)
        sam_input = torch.nn.functional.interpolate(
            sam_input, 
            size=target_sam_size, 
            mode="bilinear", 
            align_corners=False
        )
        
        # A. Backbone
        # SAM3 backbone forward expects (B, 3, H, W)
        backbone_out = self.student.backbone.forward_image(sam_input)
        
        # Manually extract features like Sam3Image._get_img_feats but simplified for BATCH
        # We assume batch size B and we process all images.
        # backbone_out['backbone_fpn'] is list of (B, C, H_feat, W_feat)
        vis_feats = backbone_out["backbone_fpn"][-self.student.num_feature_levels :]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.student.num_feature_levels :]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        
        # Flatten: NxCxHxW => HWxNxC (seq-first)
        # Here N is Batch Size (B).
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vis_pos_enc]

        # B. Prompt Encoding
        # Construct Geometric Prompt
        # Scale boxes to 1008x1008 scale
        orig_H, orig_W = images.shape[2], images.shape[3]
        scale_y = target_sam_size[0] / orig_H
        scale_x = target_sam_size[1] / orig_W
        
        # boxes is (B, 4) in [x1, y1, x2, y2]
        boxes_scaled = boxes.clone()
        boxes_scaled[:, 0] *= scale_x
        boxes_scaled[:, 1] *= scale_y
        boxes_scaled[:, 2] *= scale_x
        boxes_scaled[:, 3] *= scale_y

        if boxes_scaled.dim() == 2:
            boxes_scaled = boxes_scaled.unsqueeze(1) # (B, 1, 4)
            
        B = boxes_scaled.shape[0]
        N_boxes = boxes_scaled.shape[1]
        device = boxes_scaled.device
        
        # Prompt expects Sequence First for Embeddings and Labels: (N, B, ...)
        # But Batch First for Masks: (B, N)
        
        box_embeddings = boxes_scaled.permute(1, 0, 2) # (N, B, 4)
        box_labels = torch.ones((N_boxes, B), device=device, dtype=torch.long) # (N, B)
        box_mask = torch.zeros((B, N_boxes), device=device, dtype=torch.bool) # (B, N)
        
        # Prompt object
        geometric_prompt = Prompt(
            box_embeddings=box_embeddings, 
            box_mask=box_mask, 
            box_labels=box_labels
        )
        
        # Run geometry encoder
        geo_feats, geo_masks = self.student.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        # No Visual/Text prompts for now
        prompt = geo_feats
        prompt_mask = geo_masks

        # C. Transformer Encoder
        # _run_encoder logic
        prompt_pos_embed = torch.zeros_like(prompt)
        
        memory = self.student.transformer.encoder(
            src=[x.clone() for x in img_feats], # copy list?
            src_key_padding_mask=None,
            src_pos=[x.clone() for x in img_pos_embeds],
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=None,
        )
        
        encoder_out = {
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
        }

        # D. Transformer Decoder
        # _run_decoder logic
        out = {}
        
        # Prepare decode args
        # tgt (query embed) creation replicated from Sam3Image._run_decoder
        bs = memory["memory"].shape[1]
        query_embed = self.student.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)

        apply_dac = self.student.transformer.decoder.dac and self.training
        
        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.student.transformer.decoder(
                tgt=tgt,
                memory=memory["memory"],
                memory_key_padding_mask=memory["padding_mask"],
                pos=memory["pos_embed"],
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=apply_dac,
            )
        )
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)
        
        # Scores and Boxes
        # Sam3Image._update_scores_and_boxes
        # Simplified:
        num_o2o = hs.shape[2] # Assuming no O2M for simplicity or handle it
        if apply_dac:
             num_o2o = num_o2o // 2 # Half O2O, Half O2M
             
        # Scores
        if self.student.use_dot_prod_scoring:
             outputs_class = self.student.dot_prod_scoring(hs, prompt, prompt_mask)
        else:
             outputs_class = self.student.class_embed(hs)
             
        # Boxes
        anchor_box_offsets = self.student.transformer.decoder.bbox_embed(hs)
        # inverse_sigmoid import needed? Or implement it.
        # box_ops?
        # Let's import inverse_sigmoid from sam3.model.model_misc
        
        def inverse_sigmoid(x, eps=1e-5):
            x = x.clamp(min=0, max=1)
            x1 = x.clamp(min=eps)
            x2 = (1 - x).clamp(min=eps)
            return torch.log(x1/x2)
            
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        
        # E. Segmentation Head
        # _run_segmentation_heads logic
        if self.student.segmentation_head is not None:
            # We need activation_ckpt_wrapper?
            # Or just call it.
            seg_head_outputs = self.student.segmentation_head(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=hs, # hs contains both O2O and O2M queries
                image_ids=torch.arange(B, device=device),
                encoder_hidden_states=encoder_out["encoder_hidden_states"],
                act_ckpt_enable=False,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            # seg_head_outputs has 'pred_masks' (B, N_queries, H, W) logits.
            
            # Extract the relevant mask
            # We want the mask corresponding to our box prompt.
            # Usually Index 0??
            # SAM outputs multiple queries.
            # We need to match prediction to GT or pick Best?
            # Start with index 0 if 1 prompt.
            
            student_logits_low = seg_head_outputs["pred_masks"]
            
            # We take the first query result? 
            # Or use IoU/Score to pick?
            # For Distillation, maybe just pick the one with highest score or index 0 (as prompt-driven usually focuses on query 0)?
            # With box prompt, there might be ambiguity, but usually query 0 is good?
            
            # Pick query 0 (B, 1, H, W)
            student_logits_low = student_logits_low[:, 0:1, :, :]
            
        else:
            raise ValueError("No segmentation head!")

        # F. Post-process
        # Upsample to Image Size
        target_H, target_W = images.shape[2], images.shape[3]
        student_logits = torch.nn.functional.interpolate(
            student_logits_low,
            size=(target_H, target_W),
            mode="bilinear",
            align_corners=False
        )

        return teacher_logits, student_logits