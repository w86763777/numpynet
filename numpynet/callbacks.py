class Callback(object):
    def on_train_begin(self, model):
        pass

    def on_train_end(self, model):
        pass

    def on_epoch_begin(self, model, epoch):
        pass

    def on_epoch_end(self, model, epoch, log={}):
        pass

    def on_batch_begin(self, model, epoch, batch):
        pass

    def on_batch_end(self, model, epoch, batch, log={}):
        pass


class History(Callback):
    def on_train_begin(self, model):
        self.iteration = 0
        self.history = dict()
        for metric in model.metrics:
            self.history[metric.__name__] = []
            self.history['val_' + metric.__name__] = []

    def on_epoch_end(self, model, epoch, log={}):
        for k, v in log.items():
            self.history[k].append((self.iteration, v))

    def on_batch_begin(self, model, epoch, batch):
        self.iteration += 1

    def on_batch_end(self, model, epoch, batch, log={}):
        for k, v in log.items():
            self.history[k].append((self.iteration, v))


class ExtraValidation(History):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def on_train_begin(self, model):
        super().on_train_begin(model)
        for metric in model.metrics:
            self.history['extra_val_' + metric.__name__] = []

    def on_epoch_end(self, model, epoch, log={}):
        super().on_epoch_end(model, epoch, log)
        evals = model.evaluate(self.val.X, self.val.y)
        for v, metric in zip(evals, model.metrics):
            self.history['extra_val_' + metric.__name__].append(
                (self.iteration, v))


class Checkpoint(Callback):
    def __init__(self, path, metric):
        self.path = path
        self.metric_name = 'val_' + metric.__name__
        self.max_val = 0

    def on_epoch_end(self, model, epoch, log={}):
        if epoch == 1 or log[self.metric_name] > self.max_val:
            print('Save model with better performance: %f > %f' % (
                log[self.metric_name], self.max_val))
            self.max_val = log[self.metric_name]
            model.save(self.path)
