import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
import copy

from evaluate.mae_utils import convert_to_tensor, prepare_model
from models_mae import mae_vit_large_patch16_dec512d8b, MaskedAutoencoderViT

# the authors chose a padding of 1 in demo.ipynb.
# The input image is 224x224 px, so each grid image is 224//2x224//2 px
image_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((111, 111)), torchvision.transforms.ToTensor()]
)


class VisualPrompting:
    def __init__(self, transformer_path, device="cuda") -> None:
        # From: evaluate/mae_utils.py
        self.model: MaskedAutoencoderViT = mae_vit_large_patch16_dec512d8b()
        ckpt = torch.load(transformer_path, map_location="cpu", weights_only=False)
        msg = self.model.load_state_dict(ckpt["model"], strict=False)
        if "<All keys matched successfully>" not in msg:
            print(msg)

        self.model.to(device).eval()
        self.device = device

    def fill_to_full_batched(self, arrs):
        new_arr = copy.deepcopy(arrs)
        # if isinstance(new_arr, np.ndarray):
        new_arr = [list(n) for n in new_arr]
        for i in range(196):
            for k in new_arr:
                if i not in k:
                    k.append(i)
        return torch.tensor(new_arr)

    def obtain_values_from_mask(self, mask: np.ndarray, batch_size):
        if mask.shape == (batch_size, 14, 14):
            return [m.flatten().nonzero()[0] for m in mask]
        assert mask.shape == (batch_size, 224, 224)
        counter = 0
        values = []
        for i in range(0, 224, 16):
            for j in range(0, 224, 16):
                if np.sum(mask[i : i + 16, j : j + 16]) == 16**2:
                    values.append(counter)
                counter += 1
        return values

    def generate_mask_for_evaluation(self, batch_size):
        mask = np.zeros((batch_size, 14, 14))
        # fill top right, bottom left, top left with ones
        mask[:, :7] = 1
        mask[:, :, :7] = 1
        mask = self.obtain_values_from_mask(mask, batch_size)
        len_keep = len(mask[0])
        return self.fill_to_full_batched(mask), len_keep

    def images_to_grid(self, ctx_in, ctx_gt, prompt):
        """Image tensors of shape c,h,w"""
        assert ctx_gt.shape == ctx_in.shape and ctx_in.shape == prompt.shape
        # c,h*2+2p,w*2+2p (in our case (3,111*2+2,111*2+2)=(3,224,224))
        grid = torch.zeros(
            size=(ctx_in.shape[0], ctx_in.shape[1] * 2 + 2, ctx_in.shape[2] * 2 + 2)
        )
        grid[:, : ctx_in.shape[1], : ctx_in.shape[2]] = ctx_in
        # + 2 is because of padding!!
        grid[:, : ctx_gt.shape[1], ctx_in.shape[2] + 2 :] = ctx_gt
        grid[:, ctx_in.shape[1] + 2 :, : prompt.shape[2]] = prompt
        # Bottom right of image stays zero? TODO

        return grid

    def _model_forward(self, x, ids_shuffle, len_keep):
        """We need to reimplement forward function since apparently there is
        no option for evaluation (no random masking) in the model.
        This is inspired by mae_utils.py"""
        
        latent = self.model.patch_embed(x)
        latent = latent + self.model.pos_embed[:, 1:, :]

        N, L, D = (
            latent.shape
        )  # batch, length, dim = (N, amt_patches, pixels_per_patch)
        # sort noise for each sample
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        latent = torch.gather(
            latent, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=latent.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # append cls token
        cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(latent.shape[0], -1, -1)
        latent = torch.cat((cls_tokens, latent), dim=1)
        # apply Transformer blocks
        for blk in self.model.blocks:
            latent = blk(latent)
        latent = self.model.norm(latent)

        return mask, self.model.forward_decoder(latent, ids_restore)

    @torch.no_grad()
    def inference(self, img_batch):
        """img_batch is list of list of PIL image
        [
            [ctx_in_1, ctx_gt_1, ..., prompt] # This should only contain 3 images. If not, last 3 are used.
        ]

        Important: This model only allows for one context pair and one prompt!!
        So, inference uses only the last three images of each batch element.

        Args:
            img_batch (PIL.image): image
        """
        # Convert single images to tensors, but keep lists.
        img_batch = [[image_transform(image) for image in b] for b in img_batch]

        # Convert images in inner lists to grids.
        grids = [self.images_to_grid(im[-3], im[-2], im[-1]) for im in img_batch]
        grids = torch.stack(grids)
        # TODO     grid = (grid - imagenet_mean[:,None,None]) / imagenet_std[:, None,None]
        grids = grids.to(self.device)

        # Apparently forward_encoder() does this with self.patch_embed()
        # grids = self.model.patchify(grids)

        ids_shuffle, len_keep = self.generate_mask_for_evaluation(
            batch_size=grids.shape[0]
        )

        mask, pred = self._model_forward(grids, ids_shuffle, len_keep)
        # Predictions are raw logits pointing to codebook indices.
        pred = torch.argmax(pred, dim=-1)
        # After finding the most likely codebook index for each patch,
        # get its z_q
        pred = self.model.vae.quantize.get_codebook_entry(
            pred.reshape(-1),
            [pred.shape[0], pred.shape[-1] // 14, pred.shape[-1] // 14, -1],
        )

        # vqgan z_q -> pixels
        pred = self.model.vae.decode(pred)
        # Mask has 1 where prediction is (bottom right), else 0
        mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 * 3)
        mask = self.model.unpatchify(mask)

        pred = grids * (1 - mask) + pred * mask
        
        return pred.permute(0, 2, 3, 1)  # TODO do i have to reshape to fit h,w,c?


if __name__ == "__main__":
    import requests
    from PIL import Image
    from io import BytesIO
    import PIL

    def has_transparency(img):
        if img.info.get("transparency", None) is not None:
            return True
        if img.mode == "P":
            transparent = img.info.get("transparency", -1)
            for _, index in img.getcolors():
                if index == transparent:
                    return True
        elif img.mode == "RGBA":
            extrema = img.getextrema()
            if extrema[3][0] < 255:
                return True

        return False

    def url_to_pil(url) -> Image.Image:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    # These are from the repo
    source = "https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2022/06/14/ML-8362-image001-300.jpg"
    target = "https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2022/06/14/ML-8362-image003-300.png"
    new_source = "https://static.scientificamerican.com/sciam/cache/file/1E3A3E62-B3CA-434A-8C3B3ED0C982FB69_source.jpg?w=590&h=800&C8DB8C57-989B-4118-AE27EF1191E878A5"

    vp = VisualPrompting(
        transformer_path="./models/mae_vit_cvf+in/checkpoint-3400.pth", device="cpu"
    )
    imgs = vp.inference(
        [
            [
                url_to_pil(source).convert("RGB"),
                url_to_pil(target).convert("RGB"),
                url_to_pil(new_source).convert("RGB"),
            ],
            [
                url_to_pil(source).convert("RGB"),
                url_to_pil(target).convert("RGB"),
                url_to_pil(new_source).convert("RGB"),
            ],
        ]
    )
    print(type(imgs), type(imgs[0]), imgs[0].shape)
    PIL.Image.fromarray(
        (torch.clamp(imgs[0], 0.0, 1.0) * 255).cpu().detach().numpy().astype(np.uint8)
    ).show()
