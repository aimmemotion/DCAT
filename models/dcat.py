
"""
Modifed from https://github.com/IBM/CrossViT

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, patches, num_heads=8, select=False, keep_ratio=1.0, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.select = select
        self.keep_ratio = keep_ratio
        if select:
            self.gen_new_x = nn.Linear(int(patches * keep_ratio), patches)
        #print("patches: {}".format(patches))

    def forward(self, x, otherBranch_x):

        B, N, C = x.shape
        #print("N: {}".format(N))

        #print("x: {}".format(x.shape))

        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        if self.select:
            #print("select!")
            x_cls = x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
            cls_k = x[:, 1:, ...].reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B(N-1)C -> B(N-1)H(C/H) -> BH(N-1)(C/H)
            cls_attn = (x_cls @ cls_k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)(N-1) -> BH1(N-1)
            cls_attn = torch.mean(cls_attn, dim=1)  # BH1(N-1) -> B1(N-1)
            cls_attn = cls_attn.softmax(dim=-1)

            keep_token_num = int((N-1) * self.keep_ratio)
            score = torch.argsort(cls_attn, dim=2, descending=True)  # B1(N-1)
            
            keep_policy = score[:, :, :keep_token_num]  # B1(keep_token_num)
            keep_policy = keep_policy.reshape(B, keep_token_num)  # B1(keep_token_num) -> B(keep_token_num)
            
            keep_policy, _ = torch.sort(keep_policy, 1, descending=False)

            temp_x = x[:, 1:, ...]
            
            keep_policy = torch.unsqueeze(keep_policy, 2)
            keep_policy = keep_policy.repeat(1, 1, C)        
            
            select_x = temp_x.gather(1, keep_policy)  # B(keep_token_num)C

            #print("select_x: {}".format(select_x.shape))
            
            new_x = self.gen_new_x(select_x.transpose(-2, -1))
            new_x = new_x.transpose(-2, -1)
            new_x = torch.cat((x[:, 0:1, ...], new_x), dim=1)

            q = self.wq(new_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)


        #q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(otherBranch_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(otherBranch_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH(C/H)N -> BHNN
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # (BHNN @ BHN(C/H)) -> BHN(C/H) -> BNH(C/H) -> BNC
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, patches, num_heads, select=False, keep_ratio=1.0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, patches=patches, num_heads=num_heads, select=select, keep_ratio=keep_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        

    def forward(self, x, otherBranch_x):
        
        out = x + self.drop_path(self.attn(self.norm1(x), self.norm1(otherBranch_x)))

        return out


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, mip_select=True, mip_keep_ratio=0.5, full_select=True, full_keep_ratio=0.5,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs1 = nn.ModuleList()
        for d in range(num_branches):
            if patches[d] == patches[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(patches[d]+1), act_layer(), nn.Linear(patches[d]+1, patches[(d+1) % num_branches]+1)]
            self.projs1.append(nn.Sequential(*tmp))
            
        self.projs2 = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs2.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        
        ###
        # MIP branch
        if depth[-1] == 0:  # backward capability:
            self.fusion.append(CrossAttentionBlock(dim=dim[0], patches=patches[0], num_heads=num_heads[1], select=mip_select, keep_ratio=mip_keep_ratio,
                                                   mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                                   drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
        else:
            tmp = []
            for _ in range(depth[-1]):
                tmp.append(CrossAttentionBlock(dim=dim[0], patches=patches[0], num_heads=num_heads[1], select=mip_select, keep_ratio=mip_keep_ratio,
                                               mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                               drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
            self.fusion.append(nn.Sequential(*tmp))
        
        # Full branch
        if depth[-1] == 0:  # backward capability:
            self.fusion.append(CrossAttentionBlock(dim=dim[1], patches=patches[1], num_heads=num_heads[0], select=full_select, keep_ratio=full_keep_ratio,
                                                   mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                                   drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
        else:
            tmp = []
            for _ in range(depth[-1]):
                tmp.append(CrossAttentionBlock(dim=dim[1], patches=patches[1], num_heads=num_heads[0], select=full_select, keep_ratio=full_keep_ratio,
                                               mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                               drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
            self.fusion.append(nn.Sequential(*tmp))
        ###        
        

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        #proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        proj_ob_token = []
        proj_ob_token.append(outs_b[0])
        proj_ob_token.append(outs_b[1])
        proj_ob_token[0] = proj_ob_token[0].transpose(-2, -1)
        proj_ob_token[1] = proj_ob_token[1].transpose(-2, -1)
        proj_ob_token = [proj(x) for x, proj in zip(proj_ob_token, self.projs1)]
        proj_ob_token[0] = proj_ob_token[0].transpose(-2, -1)
        proj_ob_token[1] = proj_ob_token[1].transpose(-2, -1)
        proj_ob_token = [proj(x) for x, proj in zip(proj_ob_token, self.projs2)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            #tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](outs_b[i], proj_ob_token[(i + 1) % self.num_branches])
            #reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            #tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384), depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 mip_select=True, mip_keep_ratio=0.5, full_select=True, full_keep_ratio=0.5,
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)

        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))

            del self.pos_embed
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  mip_select=mip_select, mip_keep_ratio=mip_keep_ratio, full_select=full_select, full_keep_ratio=full_keep_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])
        
        #self.branch_weight = nn.ModuleList([nn.Linear(embed_dim[0]+embed_dim[1], 2)])
        #self.softmax3 = nn.Softmax(dim=1)

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, imgs):
        
        B, C, H, W = imgs[0].shape
        xs = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(imgs[i], size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else imgs[i]
            tmp = self.patch_embed[i](x_)
            #print("patch_embed shape: {}".format(tmp.shape))
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)
            #print("tmp.shape: {}".format(tmp.shape))

        for blk in self.blocks:
            xs = blk(xs)

        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]

        return out

    def forward(self, fullImg, mipImg):
        imgs = [mipImg, fullImg]
        xs = self.forward_features(imgs)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        
        return ce_logits



@register_model
def dcat(pretrained=False, mip_select=True, mip_keep_ratio=0.5, full_select=True, full_keep_ratio=0.5, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              mip_select=mip_select, mip_keep_ratio=mip_keep_ratio, full_select=full_select, full_keep_ratio=full_keep_ratio,
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    return model


