from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class TransferWeight(Hook):

    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class EpochStopHook(Hook):

    def __init__(self, every_n_epochs=1):
        self.interval = every_n_epochs

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            raise StopIteration("Training stopped by EpochStopHook.")
