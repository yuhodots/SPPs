import torch
import fnmatch
from torchvision.models.resnet import _resnet, Bottleneck
from timm.utils.model import avg_ch_var, avg_ch_var_residual, avg_sq_ch_mean
from matplotlib import pyplot as plt


def resnet_v2_600(pretrained=False, progress=False, **kwargs):
    return _resnet('resnet_v2_600', Bottleneck, [50, 50, 50, 50], pretrained, progress, **kwargs)


class ActivationStatsHook(object):
    def __init__(self, model, hook_fn_locs, hook_fns):
        self.model = model
        self.hook_fn_locs = hook_fn_locs
        self.hook_fns = hook_fns
        self.stats = dict((hook_fn.__name__, []) for hook_fn in hook_fns)
        for hook_fn_loc, hook_fn in zip(hook_fn_locs, hook_fns):
            self._register_hook(hook_fn_loc, hook_fn)

    def _create_hook(self, hook_fn):
        def append_activation_stats(module, input, output):
            out = hook_fn(module, input, output)
            self.stats[hook_fn.__name__].append(out)

        return append_activation_stats

    def _register_hook(self, hook_fn_loc, hook_fn):
        for name, module in self.model.named_modules():
            if not any([fnmatch.fnmatch(name, hook_loc) for hook_loc in hook_fn_loc]):
                continue
            module.register_forward_hook(self._create_hook(hook_fn))

    def extract_spp_stats(self, x=None):
        if x is None:
            input_shape = [8, 3, 224, 224]
            x = torch.normal(0., 1., input_shape)
        _ = self.model(x)
        return self.stats


class SignalPropagationPlots(object):
    def __init__(self, model, hook_fn_locs=None, hook_fns=None):
        if hook_fn_locs is None:
            hook_fn_locs = [['layer?.?', 'layer?.??'],
                            ['layer?.?', 'layer?.??'],
                            ['layer?.?.bn3', 'layer?.??.bn3']]
        if hook_fns is None:
            hook_fns = [avg_sq_ch_mean, avg_ch_var, avg_ch_var_residual]

        self.hook = ActivationStatsHook(model, hook_fn_locs, hook_fns)

    def get_stats(self, x=None):
        return self.hook.extract_spp_stats(x)

    @staticmethod
    def plot_stats(stats, save=False, save_path="./spp.png"):
        fig, ax = plt.subplots(1, 3, figsize=(18, 3), sharey=True)

        # Plot
        ax[0].plot(stats['avg_sq_ch_mean'], label='avg_sq_ch_mean')
        ax[0].set_ylabel('Average Square Channel Mean')
        ax[0].grid()

        ax[1].plot(stats['avg_ch_var'], label='avg_ch_var')
        ax[1].set_ylabel('Average Channel Variance')
        ax[1].grid()

        ax[2].plot(stats['avg_ch_var_residual'], label='avg_ch_var_residual')
        ax[2].set_ylabel('Residual Average Channel Variance')
        ax[2].grid()

        # Save
        if save:
            plt.savefig(save_path)

        plt.show()


def main():
    model = resnet_v2_600()
    spp = SignalPropagationPlots(model)
    stats = spp.get_stats()
    spp.plot_stats(stats, save=True, save_path="assets/img/spp.png")


if __name__ == "__main__":
    main()
