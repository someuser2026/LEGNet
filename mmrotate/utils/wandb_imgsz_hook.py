from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class WandbImgszHook(Hook):
    """Log image size metadata once to W&B.

    This hook expects WandbLoggerHook to be enabled in ``log_config``.
    """

    def __init__(self, imgsz=None, log_key='imgsz'):
        self.imgsz = imgsz
        self.log_key = log_key
        self._logged = False

    def after_train_iter(self, runner):
        # Log once after W&B has already been initialized by logger hooks.
        if self._logged or runner.iter != 0:
            return
        self._log_to_wandb(runner)

    def _log_to_wandb(self, runner):
        if self.imgsz is None:
            self._logged = True
            return

        wandb_hook = None
        for hook in runner.hooks:
            if hook.__class__.__name__ == 'WandbLoggerHook':
                wandb_hook = hook
                break

        if wandb_hook is None:
            self._logged = True
            return

        wandb = getattr(wandb_hook, 'wandb', None)
        if wandb is None or getattr(wandb, 'run', None) is None:
            return

        value = list(self.imgsz) if isinstance(self.imgsz, tuple) else self.imgsz
        payload = {self.log_key: value}

        try:
            wandb.config.update(payload, allow_val_change=True)
        except TypeError:
            wandb.config.update(payload)

        wandb.log(payload, step=runner.iter + 1)
        self._logged = True
