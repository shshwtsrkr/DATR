import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import DecoderLayer, Decoder, Encoder, TokenEmbedding, StageDecoder, StageEncoder

import torch.distributed as dist
import clip
from functools import partial


class BaseSeq(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length

        self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio, global_pool='token', class_token=True)
        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        
        tgt = tgt.long()  

        
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        
        num_steps = max_length + 1
        memory = self.encode(images)
        if isinstance(memory, list):
            memory = memory[-1]

        
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = self.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  
            
            
            
            
            tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                  tgt_query_mask=query_mask[i:j, :j])
            if isinstance(tgt_out, list):
                tgt_out = tgt_out[-1]
            
            p_i = self.head(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                
                if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)

        return logits

    def gen_tgt_perms(self, tgt):
        max_num_chars = tgt.shape[1] - 2
        
        if max_num_chars == 1:
            return torch.arange(3, device=self._device)
        perm = torch.arange(max_num_chars, device=self._device)
        bos_idx = perm.new_zeros((1,))
        eos_idx = perm.new_full((1,), max_num_chars + 1)
        perm = torch.cat([bos_idx, perm + 1, eos_idx], dim=0)

        return perm

    def generate_attn_masks(self, perm):
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        
        memory = self.encode(images)

        
        perm = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        tgt_mask, query_mask = self.generate_attn_masks(perm)
        out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = self.head(out).flatten(end_dim=1)
        loss = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)

        self.log('loss', loss)
        return loss


class DATR(BaseSeq):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, max_label_length,
                         batch_size, lr, warmup_pct, weight_decay,
                         img_size, patch_size, embed_dim,
                         enc_num_heads, enc_mlp_ratio, enc_depth,
                         dec_num_heads, dec_mlp_ratio, dec_depth,
                         dropout, **kwargs)

        self.encoder = StageEncoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
                               mlp_ratio=enc_mlp_ratio, global_pool='token', class_token=True)

        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = StageDecoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)

        
        self.align_head0 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512, eps=1e-5),
        )
        self.align_head1 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512, eps=1e-5),
        )
        self.align_head2 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512, eps=1e-5),
        )
        
        self.norm_head0 = nn.LayerNorm(512, eps=1e-5)
        self.norm_head1 = nn.LayerNorm(512, eps=1e-5)
        self.norm_head2 = nn.LayerNorm(512, eps=1e-5)
        
        self.align_proj0 = nn.Parameter(torch.ones((1, 17, 129)) / 129)
        self.align_head_vis0 = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.LayerNorm(768, eps=1e-5),
        )
        self.align_proj1 = nn.Parameter(torch.ones((1, 17, 129)) / 129)
        self.align_head_vis1 = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.LayerNorm(768, eps=1e-5),
        )
        self.align_proj2 = nn.Parameter(torch.ones((1, 17, 129)) / 129)
        self.align_head_vis2 = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.LayerNorm(768, eps=1e-5),
        )
        
        self.norm_head_vis0 = nn.LayerNorm(768, eps=1e-5)
        self.norm_head_vis1 = nn.LayerNorm(768, eps=1e-5)
        self.norm_head_vis2 = nn.LayerNorm(768, eps=1e-5)
        
        
        self.word_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 512))

        named_apply(partial(init_weights, exclude=['encoder']), self)

        self.clip = clip.load("ViT-B/16", device='cpu')
        self.helper_clip = clip.load("RN50x16", device='cpu')

        
        positional_embedding = self.clip.visual.positional_embedding.data
        assert positional_embedding.shape[0] == 14 * 14 + 1
        positional_embedding = torch.cat((positional_embedding[:8 + 1], positional_embedding[14 + 1:22 + 1]), dim=0)
        self.clip.visual.positional_embedding = torch.nn.Parameter(positional_embedding)
        for p in self.clip.parameters():
            p.requires_grad = False

        self.clip_mean = nn.Parameter(torch.tensor((0.48145466, 0.4578275, 0.40821073)).reshape(1,3,1,1))
        self.clip_mean.requires_grad = False
        self.clip_std = nn.Parameter(torch.tensor((0.26862954, 0.26130258, 0.27577711)).reshape(1,3,1,1))
        self.clip_std.requires_grad = False

        
        self.helper_clip_mean = nn.Parameter(torch.tensor((0.48145466, 0.4578275, 0.40821073)).reshape(1,3,1,1))
        self.helper_clip_mean.requires_grad = False
        self.helper_clip_std = nn.Parameter(torch.tensor((0.26862954, 0.26130258, 0.27577711)).reshape(1,3,1,1))
        self.helper_clip_std.requires_grad = False

        self.align_loss = LCL
    
    def similar_cosine_batch(self, images):
        with torch.no_grad():
            images_clip = images * 0.5 + 0.5
            images_clip = (images_clip - self.clip_mean) / self.clip_std
            image_features = self.clip.encode_image(images_clip)
            norm_features = image_features / image_features.norm(dim=1, keepdim=True)
            similarity_matrix = torch.mm(norm_features, norm_features.transpose(0, 1))
            similarity_matrix.fill_diagonal_(-1)
            _, indices = similarity_matrix.max(dim=1)
            similar_images = images[indices]
        return similar_images, indices

    def preprocess_images(self, images):
        def enhance_contrast(image, factor=1.5):
            mean = torch.mean(image)
            enhanced_image = (1 - factor) * mean + factor * image
            return enhanced_image
        def sharpen_image(image):
            sharpening_kernel_single = torch.tensor([[-1, -1, -1],
                                                    [-1, 9, -1],
                                                    [-1, -1, -1]], dtype=torch.float32)
            sharpening_kernel = sharpening_kernel_single.repeat(3, 1, 1).unsqueeze(1)   
            sharpening_kernel = sharpening_kernel.to(image.device)
            sharpened_image = F.conv2d(image.unsqueeze(0), sharpening_kernel, padding=1, groups=3)
            return sharpened_image.squeeze(0)
        processed_images = []
        for image in images:
            sharpened = sharpen_image(image)
            enhanced = enhance_contrast(sharpened)
            processed_images.append(enhanced)
        processed_images = torch.stack(processed_images)
        return processed_images

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        images1, images2 = torch.tensor_split(images, 2)
        images1 = self.preprocess_images(images1)
        n = int(len(labels)/2)
        labels1 = labels[:n]
        labels2 = labels[n:]
        images1_flat = images1.view(images1.size(0), -1)
        images2_flat = images2.view(images2.size(0), -1)
        similarity = torch.matmul(images1_flat, images2_flat.T)
        similarity = F.cosine_similarity(images1_flat[:, None, :], images2_flat[None, :, :], dim=-1)
        _, indices = torch.max(similarity, dim=1)
        labels2 = [labels2[i] for i in indices]
        images2 = images2[indices]
        
        #HELPER CLIP
        helper_tgt = self.tokenizer.encode(labels2, self._device)

        with torch.no_grad():
            helper_labels_clip = [' '.join(w) for w in labels2]
            helper_text = clip.tokenize(helper_labels_clip, context_length=helper_tgt.shape[1]).to(images2.device)
            self.helper_clip.eval()
            helper_clip_list = self.helper_clip.encode_text(helper_text, return_list=True)  
            helper_f_list = helper_clip_list[:-1][::-1]
            helper_clip_embed = helper_clip_list[-1]

            helper_images_clip = images2 * 0.5 + 0.5
            helper_images_clip = (helper_images_clip-self.helper_clip_mean)/self.helper_clip_std
            helper_f_list_vis = self.helper_clip.encode_image(helper_images_clip, return_list=True)[:-1]  

        #PRIMARY CLIP
        tgt = self.tokenizer.encode(labels1, self._device)

        with torch.no_grad():
            primary_labels_clip = [' '.join(w) for w in labels1]
            primary_text = clip.tokenize(primary_labels_clip, context_length=tgt.shape[1]).to(images.device)
            self.clip.eval()
            primary_clip_list = self.clip.encode_text(primary_text, return_list=True)
            primary_f_list = primary_clip_list[:-1][::-1]
            primary_clip_embed = primary_clip_list[-1]
            
            primary_images_clip = images1 * 0.5 + 0.5
            primary_images_clip = (primary_images_clip-self.clip_mean)/self.clip_std
            primary_f_list_vis = self.clip.encode_image(primary_images_clip, return_list=True)[:-1]  


        
        
        """ 
            M = Modified; P = Primary
            CE = CLIP EMBED; FL = F_LIST (CLIP TEXT ENCODER); FLV = F_LIST_VIS(CLIP IMAGE ENCODER)

            WORKING COMBINATIONS:
                1. MCE + MFL + MFLV
                2. MCE + PFL + MFLV
                3. MCE + PFL + PFLV
                4. MCE + MFL + PFLV
                5. PCE + PFL + PFLV
                6. PCE + MFL + MFLV
                7. PCE + PFL + MFLV
                8. PCE + MFL + PFLV
        """
        
        #MCE
        # linear_transform = nn.Linear(in_features=helper_clip_embed.shape[-1], out_features=primary_clip_embed.shape[-1]).to('cuda')
        # clip_embed = linear_transform(helper_clip_embed)

        #PCE
        clip_embed = primary_clip_embed
        
        #PFL
        # f_list = primary_f_list

        #MFL
        for primary_f, helper_f in zip(primary_f_list, helper_f_list):
            linear_transform_f_list = nn.Linear(in_features=helper_f.shape[-1], out_features=primary_f.shape[-1]).to('cuda')
            helper_f = linear_transform_f_list(helper_f)
            f_list=[(torch.cat([primary_f, helper_f], dim=1))]

        #MFLV
        # f_list_vis = [torch.cat([helper_f_vis, primary_f_vis], dim=1) for helper_f_vis, primary_f_vis in zip(helper_f_list_vis, primary_f_list_vis)]

        #PFLV
        f_list_vis = primary_f_list_vis
        
        memory_list = self.encode(images1)

        perm = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        tgt_mask, query_mask = self.generate_attn_masks(perm)
        out_list = self.decode(tgt_in, memory_list[-1], tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)

        loss_align = 0.0
        for i, (seq1, seq2) in enumerate(zip(out_list[:-1], f_list)):
            seq1 = eval("self.align_head" + str(i))(seq1.flatten(0, -2))
            seq2 = eval("self.norm_head" + str(i))(seq2[:, 1:].flatten(0, -2))
            seq1_len, seq2_len = seq1.shape[0], seq2.shape[0]
            if seq1_len > seq2_len:
                
                padding = seq1_len - seq2_len
                seq2 = F.pad(seq2, (0, 0, 0, padding), "constant", 0)
            elif seq2_len > seq1_len:
                
                padding = seq2_len - seq1_len
                seq1 = F.pad(seq1, (0, 0, 0, padding), "constant", 0)
            loss_align = loss_align + self.align_loss(seq1, seq2)

        loss_align_vis = 0.0
        for i, (seq1, seq2) in enumerate(zip(memory_list[:-1], f_list_vis)):
            seq1 = torch.matmul(eval("self.align_proj" + str(i)).expand(seq1.shape[0],-1,-1), seq1)
            seq1 = eval("self.align_head_vis" + str(i))(seq1.flatten(0, -2))
            seq2 = eval("self.norm_head_vis" + str(i))(seq2.flatten(0, -2))
            seq1_len, seq2_len = seq1.shape[0], seq2.shape[0]
            if seq1_len > seq2_len:
                
                padding = seq1_len - seq2_len
                seq2 = F.pad(seq2, (0, 0, 0, padding), "constant", 0)  
            elif seq2_len > seq1_len:
                
                padding = seq2_len - seq1_len
                seq1 = F.pad(seq1, (0, 0, 0, padding), "constant", 0)  

            loss_align_vis = loss_align_vis + self.align_loss(seq1, seq2)

        word_embed = self.word_embed(memory_list[-1][:, 0:1]).squeeze(1)
        loss_word = self.align_loss(word_embed, clip_embed)

        logits = self.head(out_list[-1]).flatten(end_dim=1)
        loss_char = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)

        loss = loss_char + loss_word + loss_align + loss_align_vis
        self.log('loss', loss)
        return loss


def LCL(image_features, text_features, cons_weight=0.1, l1_weight=5.0, tau=0.03):
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True)+1e-7)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-7)
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        length = torch.tensor(text_features.shape[0],device=text_features.device)
        gathered_length = [torch.zeros_like(length) for _ in range(world_size) ]
        dist.all_gather(gathered_length, length)
        max_length = max(gathered_length).item()

        extend_text_features = torch.zeros((max_length,text_features.shape[1]),device=text_features.device)
        extend_text_features[:text_features.shape[0]] += text_features
        gathered_text_features = [
             torch.zeros_like(extend_text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_text_features, extend_text_features)
        gathered_text_features = [gathered_text_features[i][:l.item()] for i, l in enumerate(gathered_length)]

        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )
        logits_per_image = image_features @ all_text_features.t()
    else:
        logits_per_image = image_features @ text_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

    loss1 = cons_weight * F.cross_entropy(logits_per_image / tau, ground_truth)

    image_similar = image_features @ image_features.t()
    text_similar = text_features @ text_features.t()
    loss2 = l1_weight * F.l1_loss(image_similar, text_similar)

    return loss1 + loss2