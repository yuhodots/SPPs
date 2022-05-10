import torch
import fnmatch
from torchvision.models.resnet import _resnet, Bottleneck
from matplotlib import pyplot as plt
from celluloid import Camera


# Model used in Brock et al.,
def resnet_v2_600(pretrained=False, progress=False, **kwargs):
    return _resnet('resnet_v2_600', Bottleneck, [50, 50, 50, 50], pretrained, progress, **kwargs)


# https://gist.github.com/amaarora/6ff337e2823f06ac74a88ac03b5e2576
def avg_sq_ch_mean(model, input, output):
    """ calculate average channel square mean of output activations """
    return torch.mean(output.mean(axis=[0, 2, 3])**2).item()


def avg_ch_var(model, input, output):
    """ calculate average channel variance of output activations """
    return torch.mean(output.var(axis=[0, 2, 3])).item()


def avg_ch_var_residual(model, input, output):
    """ calculate average channel variance of output activations """
    return torch.mean(output.var(axis=[0, 2, 3])).item()


# https://gist.github.com/amaarora/2c6199c3441c0d72f356f39fb9f59611
class ActivationStatsHook(object):
    def __init__(self, model, hook_fn_locs, hook_fns):
        self.model = model
        self.hook_fn_locs = hook_fn_locs
        self.hook_fns = hook_fns
        self.stats = dict((hook_fn.__name__, []) for hook_fn in hook_fns)
        self.info = dict((hook_fn.__name__, []) for hook_fn in hook_fns)
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
            self.info[hook_fn.__name__].append(name)

    def clear_stats(self):
        hook_fns = self.hook_fns
        self.stats = dict((hook_fn.__name__, []) for hook_fn in hook_fns)

    def extract_spp_stats(self, x=None):
        if x is None:
            input_shape = [8, 3, 224, 224]
            x = torch.normal(0., 1., input_shape)
        _ = self.model(x)
        return self.stats


class ActivationStatsAnimation(object):
    def __init__(self, y_max=30, fps=30):
        self.log = dict()
        self.y_max = y_max
        self.fps = fps

    def write(self, stats, step):
        self.log[step] = stats

    def _make_fig(self, ax, data):
        avg_sq_ch_mean, avg_ch_var, avg_ch_var_residual = data

        ax[0].clear()
        ax[0].plot(avg_sq_ch_mean, label="avg_sq_ch_mean")
        ax[0].set_ylabel('Average Square Channel Mean')
        ax[0].grid()

        ax[1].clear()
        ax[1].plot(avg_ch_var, label="avg_ch_var")
        ax[1].set_ylabel('Average Channel Variance')
        ax[1].grid()

        ax[2].clear()
        ax[2].plot(avg_ch_var_residual, label="avg_ch_var_residual")
        ax[2].set_ylabel('Residual Average Channel Variance')
        ax[2].grid()

    def save(self, save_path):
        assert save_path is not None, "There is no `save_path`"
        fig, ax = plt.subplots(1, 3, figsize=(18, 3), sharey=True)
        camera = Camera(fig)

        for k in self.log.keys():
            data = (self.log[k]['avg_sq_ch_mean'],
                    self.log[k]['avg_ch_var'],
                    self.log[k]['avg_ch_var_residual'])
            self._make_fig(ax, data)
            camera.snap()

        animation = camera.animate(interval=50, blit=True)
        animation.save(save_path)


class SignalPropagationPlots(object):
    def __init__(self, model, hook_fn_locs=None, hook_fns=None):
        if hook_fn_locs is None:
            hook_fn_locs = [['layer?.?', 'layer?.??'],  # avg_sq_ch_mean (e.g. layer1.0 ~ layer1.49)
                            ['layer?.?', 'layer?.??'],  # avg_ch_var (e.g. layer1.0 ~ layer1.49)
                            ['layer?.?.bn3', 'layer?.??.bn3']]  # avg_ch_var_residual (e.g. layer1.0.bn3 ~ layer1.49.bn3)
        if hook_fns is None:
            hook_fns = [avg_sq_ch_mean, avg_ch_var, avg_ch_var_residual]

        self.hook = ActivationStatsHook(model, hook_fn_locs, hook_fns)
        self.anim = ActivationStatsAnimation()

    def _hook_info(self, indent=20):
        for key, value in self.hook.info.items():
            print(f"{key:<{indent}}: {value}")

    def summary(self):
        self._hook_info()

    def stats(self, x=None):
        self.hook.clear_stats()
        return self.hook.extract_spp_stats(x)

    @staticmethod
    def _make_plot(stats):
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

    def plot(self, stats):
        self._make_plot(stats)
        plt.show()

    def save(self, stats, save_path):
        assert save_path is not None, "There is no `save_path`"
        self._make_plot(stats)
        plt.savefig(save_path)


def main():
    # Configuration
    model = resnet_v2_600()
    img_save_path = "assets/img/spp.png"
    mp4_save_path = "assets/img/spp.mp4"
    x = torch.normal(0., 1., [8, 3, 224, 224])

    # Initialize SPPs
    spp = SignalPropagationPlots(model)
    spp.summary()

    # Save images
    stats = spp.stats(x)
    spp.save(stats, save_path=img_save_path)

    # Save video
    for step in range(10):
        stats = spp.stats(x)
        spp.anim.write(stats, step)
    spp.anim.save(save_path=mp4_save_path)


if __name__ == "__main__":
    main()
