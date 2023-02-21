import torch
from torch.nn import functional as F
import numpy as np
from os import path as osp

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img


@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):

    def test(self):
    # def test(self, dataset_name, img_name):
        # pad to multiplication of window_size
        with_metrics = self.opt['val'].get('metrics') is not None
        metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        if with_metrics:
            metric_results = {metric: 0 for metric in metric_results}

        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
                # Yt_hat = np.array([self.net_g(img).data.cpu().numpy() for _ in range(100)]).squeeze()
                
                for i in range(100):
                    if i == 0:
                        y = np.array([self.net_g(img).data.cpu().numpy()]).squeeze(0)
                        Yt_hat = y
                    else:
                        y = np.array([self.net_g(img).data.cpu().numpy()]).squeeze(0)
                        Yt_hat = np.concatenate((Yt_hat, y), axis=0)
                
                    img_save_pth = osp.join(self.opt['path']['visualization'], dataset_name,
                                            f'{img_name}_{i}.png')
                    y_img = tensor2img([torch.from_numpy(y)])
                
                    imwrite(y_img, img_save_pth)
                    metric_data = dict()
                    metric_data['img'] = y_img
                    metric_data['img2'] = tensor2img(self.gt.detach().cpu())
                
                    if with_metrics:
                        for name, opt_ in self.opt['val']['metrics'].items():
                            metric_results[name] = calculate_metric(metric_data, opt_)
                            print(f'{img_name}_{i}_{name}:', metric_results[name])
                
                output = Yt_hat.mean(0)
                output = np.expand_dims(output, axis=0)
                self.output = torch.from_numpy(output)

            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
