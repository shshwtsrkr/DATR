# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
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
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
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
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)
        if isinstance(memory, list):
            memory = memory[-1]

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
        tgt_in[:, 0] = self.bos_id

        logits = []
        for i in range(num_steps):
            j = i + 1  # next token index
            # Efficient decoding:
            # Input the context up to the ith token. We use only one query (at position = i) at a time.
            # This works because of the lookahead masking effect of the canonical (forward) AR context.
            # Past tokens have no access to future tokens, hence are fixed once computed.
            tgt_out = self.decode(tgt_in[:, :j], memory, tgt_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                  tgt_query_mask=query_mask[i:j, :j])
            if isinstance(tgt_out, list):
                tgt_out = tgt_out[-1]
            # the next token probability is in the output's ith token position
            p_i = self.head(tgt_out)
            logits.append(p_i)
            if j < num_steps:
                # greedy decode. add the next token index to the target input
                tgt_in[:, j] = p_i.squeeze().argmax(-1)
                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)

        return logits

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device)
        perm = torch.arange(max_num_chars, device=self._device)
        bos_idx = perm.new_zeros((1,))
        eos_idx = perm.new_full((1,), max_num_chars + 1)
        perm = torch.cat([bos_idx, perm + 1, eos_idx], dim=0)

        return perm

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        # Prepare the target sequences (input and output)
        perm = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        tgt_mask, query_mask = self.generate_attn_masks(perm)
        out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = self.head(out).flatten(end_dim=1)
        loss = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)

        self.log('loss', loss)
        return loss

class CLIPOCR(BaseSeq):
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

        #SDS
        self.align_head0 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-5),
            nn.ReLU()
        )
        self.align_head1 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-5),
            nn.ReLU()
        )
        self.align_head2 = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, eps=1e-5),
            nn.ReLU()
        )
        
        self.norm_head0 = nn.LayerNorm(512, eps=1e-5)
        self.norm_head1 = nn.LayerNorm(512, eps=1e-5)
        self.norm_head2 = nn.LayerNorm(512, eps=1e-5)
        
        self.align_proj0 = nn.Parameter(torch.ones((1, 17, 129)) / 129)
        self.align_head_vis0 = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768, eps=1e-5),
            nn.ReLU()
        )
        self.align_proj1 = nn.Parameter(torch.ones((1, 17, 129)) / 129)
        self.align_head_vis1 = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768, eps=1e-5),
            nn.ReLU()
        )
        self.align_proj2 = nn.Parameter(torch.ones((1, 17, 129)) / 129)
        self.align_head_vis2 = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.LayerNorm(768, eps=1e-5),
            nn.ReLU()
        )
        
        self.norm_head_vis0 = nn.LayerNorm(768, eps=1e-5)
        self.norm_head_vis1 = nn.LayerNorm(768, eps=1e-5)
        self.norm_head_vis2 = nn.LayerNorm(768, eps=1e-5)
        
        # More complex word embedding network
        self.word_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.ReLU(),
            nn.Linear(embed_dim, 1024),  # Increased size
            nn.ReLU(),
            nn.Dropout(0.5),  # Added dropout for regularization
            nn.Linear(1024, 512)
        )

        named_apply(partial(init_weights, exclude=['encoder']), self)

        self.clip = clip.load("ViT-B/16", device='cpu')
        self.helper_clip = clip.load("ViT-B/16", device='cpu')

        # self.clip = clip.load("path/to/ViT-B-16.pt", device='cpu')
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

        #HELPER_BATCH PARAMS
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

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        images1 = images
        labels1 = labels
        images2, indices = self.similar_cosine_batch(images1)
        indices_list = indices.tolist()
        labels2 = [labels[i] for i in indices_list]

        print(f"images1 type: {type(images1)}, shape/length: {len(images1) if isinstance(images1, list) else images1.shape}")
        print(f"labels1 type: {type(labels1)}, shape/length: {len(labels1) if isinstance(labels1, list) else labels1.shape}")
        print(f"images2 type: {type(images2)}, shape/length: {len(images2) if isinstance(images2, list) else images2.shape}")
        print(f"labels2 type: {type(labels2)}, shape/length: {len(labels2) if isinstance(labels2, list) else labels2.shape}")

        #HELPER BATCH
        helper_tgt = self.tokenizer.encode(labels2, self._device)

        with torch.no_grad():
            helper_labels_clip = [' '.join(w) for w in labels2]
            helper_text = clip.tokenize(helper_labels_clip, context_length=helper_tgt.shape[1]).to(images2.device)
            self.clip.eval()
            helper_clip_list = self.clip.encode_text(helper_text, return_list=True)  # N, 512
            helper_f_list = helper_clip_list[:-1][::-1]
            helper_clip_embed = helper_clip_list[-1]

            helper_images_clip = images2 * 0.5 + 0.5
            helper_images_clip = (helper_images_clip-self.helper_clip_mean)/self.helper_clip_std
            helper_f_list_vis = self.clip.encode_image(helper_images_clip, return_list=True)[:-1]  # N, 512

        # print(f"Type of helper f_list: {type(helper_f_list)}")
        # print(f"Type of helper clip_embed: {type(helper_clip_embed)}")
        # print(f"Type of helper f_list_vis: {type(helper_f_list_vis)}")


        #PRIMARY BATCH
        primary_tgt = self.tokenizer.encode(labels1, self._device)

        with torch.no_grad():
            primary_labels_clip = [' '.join(w) for w in labels1]
            primary_text = clip.tokenize(primary_labels_clip, context_length=primary_tgt.shape[1]).to(images.device)
            self.helper_clip.eval()
            primary_clip_list = self.helper_clip.encode_text(primary_text, return_list=True)  # N, 512
            primary_f_list = primary_clip_list[:-1][::-1]
            primary_clip_embed = primary_clip_list[-1]

            primary_images_clip = images1 * 0.5 + 0.5
            primary_images_clip = (primary_images_clip-self.clip_mean)/self.clip_std
            primary_f_list_vis = self.helper_clip.encode_image(primary_images_clip, return_list=True)[:-1]  # N, 512

        # print(f"Type of primary f_list: {type(primary_f_list)}")
        # print(f"Type of primary clip_embed: {type(primary_clip_embed)}")
        # print(f"Type of primary f_list_vis: {type(primary_f_list_vis)}")
        print(f"helper_tgt type: {type(helper_tgt)}, shape/length: {len(helper_tgt) if isinstance(helper_tgt, list) else helper_tgt.shape}")
        print(f"primary_tgt type: {type(primary_tgt)}, shape/length: {len(primary_tgt) if isinstance(primary_tgt, list) else primary_tgt.shape}")
        print(f"helper_clip_embed type: {type(helper_clip_embed)}, shape/length: {len(helper_clip_embed) if isinstance(helper_clip_embed, list) else helper_clip_embed.shape}")
        print(f"primary_clip_embed type: {type(primary_clip_embed)}, shape/length: {len(primary_clip_embed) if isinstance(primary_clip_embed, list) else primary_clip_embed.shape}")
        print(f"helper_f_list type: {type(helper_f_list)}, shape/length: {len(helper_f_list) if isinstance(helper_f_list, list) else helper_f_list.shape}")
        print(f"primary_f_list type: {type(primary_f_list)}, shape/length: {len(primary_f_list) if isinstance(primary_f_list, list) else primary_f_list.shape}")
        print(f"helper_f_list_vis type: {type(helper_f_list_vis)}, shape/length: {len(helper_f_list_vis) if isinstance(helper_f_list_vis, list) else helper_f_list_vis.shape}")
        print(f"primary_f_list_vis type: {type(primary_f_list_vis)}, shape/length: {len(primary_f_list_vis) if isinstance(primary_f_list_vis, list) else primary_f_list_vis.shape}\n\n")

        # print(f"helper_clip_embed type: {type(helper_clip_embed)}, shape/length: {len(helper_clip_embed) if isinstance(helper_clip_embed, list) else helper_clip_embed.shape}")
        # print(f"helper_f_list_vis type: {type(helper_f_list_vis)}, shape/length: {len(helper_f_list_vis) if isinstance(helper_f_list_vis, list) else helper_f_list_vis.shape}")
        # print(f"helper_f_list contents: {helper_f_list}")

        # for i, tensor in enumerate(primary_f_list):
        #     print(f"Primary Tensor {i} length (first dimension): {tensor.size(0)}")

        tgt = torch.cat([helper_tgt, primary_tgt], dim=1)
        clip_embed = torch.cat([helper_clip_embed, primary_clip_embed], dim=1)
        f_list = [torch.cat([helper_f, primary_f], dim=1) for helper_f, primary_f in zip(helper_f_list, primary_f_list)]
        f_list_vis = [torch.cat([helper_f_vis, primary_f_vis], dim=1) for helper_f_vis, primary_f_vis in zip(helper_f_list_vis, primary_f_list_vis)]

        #CONCATENATE FEATURES FROM BOTH THE CLIPS
        # helper_f_list_tensor = torch.tensor(helper_f_list)
        # primary_f_list_tensor = torch.tensor(primary_f_list, dtype=torch.float32)
        # f_list = []
        # for k in range(len(helper_f_list)):
        #     f_list.append(torch.cat((helper_f_list[k], primary_f_list[k]), dim=1))
        # clip_embed = torch.cat((helper_clip_embed, primary_clip_embed), dim=1)
        # f_list_vis = []
        # for k in range(len(helper_f_list_vis)):
        #     f_list_vis.append(torch.cat((helper_f_list_vis[k], primary_f_list_vis[k]), dim=1))

        # print(f"f_list type: {type(f_list)}, shape/length: {len(f_list) if isinstance(f_list, list) else f_list.shape}")
        # print(f"helper_f_list contents: {f_list}")
            
        for i, tensor in enumerate(f_list):
            print(f"F List Tensor {i} length (first dimension): {tensor.size(0)}")

        print(f"f_list type: {type(f_list)}, shape/length: {len(f_list) if isinstance(f_list, list) else f_list.shape}")
        print(f"f_list_vis type: {type(f_list_vis)}, shape/length: {len(f_list_vis) if isinstance(f_list_vis, list) else f_list_vis.shape}")

        # f_list_tensor = torch.cat((helper_f_list, primary_f_list), dim=1)
        # f_list = f_list_tensor.tolist()
        # clip_embed = torch.cat((helper_clip_embed, primary_clip_embed), dim=1)
        # helper_f_list_vis_tensor = torch.tensor(helper_f_list_vis, dtype=torch.float32)
        # primary_f_list_vis_tensor = torch.tensor(primary_f_list_vis, dtype=torch.float32)
        # f_list_tensor = torch.cat((helper_f_list_vis_tensor, primary_f_list_vis_tensor), dim=1)
        # f_list_vis = f_list_tensor.tolist()

        # Encode the source sequence (i.e. the image codes)
        memory_list = self.encode(images)

        # Prepare the target sequences (input and output)
        perm = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
        tgt_mask, query_mask = self.generate_attn_masks(perm)
        out_list = self.decode(tgt_in, memory_list[-1], tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)

        loss_align = 0.0
        for i, (seq1, seq2) in enumerate(zip(out_list[:-1], f_list)):
            seq1 = eval("self.align_head" + str(i))(seq1.flatten(0, -2))
            seq2 = eval("self.norm_head" + str(i))(seq2[:, 1:].flatten(0, -2))
            print(f"Sequence 1 type: {type(seq1)}, shape/length: {len(seq1) if isinstance(seq1, list) else seq1.shape}")
            print(f"Sequence 2 type: {type(seq2)}, shape/length: {len(seq2) if isinstance(seq2, list) else seq2.shape}")
            loss_align = loss_align + self.align_loss(seq1, seq2)

        loss_align_vis = 0.0
        for i, (seq1, seq2) in enumerate(zip(memory_list[:-1], f_list_vis)):
            seq1 = torch.matmul(eval("self.align_proj" + str(i)).expand(seq1.shape[0],-1,-1), seq1)
            seq1 = eval("self.align_head_vis" + str(i))(seq1.flatten(0, -2))
            seq2 = eval("self.norm_head_vis" + str(i))(seq2.flatten(0, -2))
            loss_align_vis = loss_align_vis + self.align_loss(seq1, seq2)

        word_embed = self.word_embed(memory_list[-1][:, 0:1]).squeeze(1)
        loss_word = self.align_loss(word_embed, clip_embed)

        logits = self.head(out_list[-1]).flatten(end_dim=1)
        loss_char = F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)

        # print('loss_char:{}'.format(loss_char.data),
        #       'loss_word:{}'.format(loss_word.data),
        #       'loss_align:{}'.format(loss_align.data),
        #       'loss_align_vis:{}'.format(loss_align_vis))
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
        # min_size_feature = min(image_features.size(0), text_features.size(0))
        # adjusted_image_similar = image_features[:min_size_feature, :min_size_feature]
        # adjusted_text_similar = text_features[:min_size_feature, :min_size_feature]
        logits_per_image = image_features @ all_text_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

    loss1 = cons_weight * F.cross_entropy(logits_per_image / tau, ground_truth)

    image_similar = image_features @ image_features.t()
    text_similar = text_features @ text_features.t()

    # print(f"image-similar type: {type(image_similar)}, shape/length: {len(image_similar) if isinstance(image_similar, list) else image_similar.shape}")
    # print(f"text-similar type: {type(text_similar)}, shape/length: {len(text_similar) if isinstance(text_similar, list) else text_similar.shape}")

    # min_size = min(image_similar.size(0), text_similar.size(0))
    # adjusted_image_similar = image_similar[:min_size, :min_size]
    # adjusted_text_similar = text_similar[:min_size, :min_size]

    # print(f"adjusted_image-similar type: {type(adjusted_image_similar)}, shape/length: {len(adjusted_image_similar) if isinstance(adjusted_image_similar, list) else adjusted_image_similar.shape}")
    # print(f"adjusted_text-similar type: {type(adjusted_text_similar)}, shape/length: {len(adjusted_text_similar) if isinstance(adjusted_text_similar, list) else adjusted_text_similar.shape}")

    loss2 = l1_weight * F.l1_loss(image_similar, text_similar)

    # print('loss1:{}, loss2:{}'.format(loss1, loss2))
    return loss1 + loss2