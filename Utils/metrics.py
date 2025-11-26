import numpy as np
import torch
import torchmetrics.classification
from torchmetrics.detection import MeanAveragePrecision
import torchmetrics


class IdentityMetrics(object):
    def __init__(self):
        self.default_metric = 'map50'
        pass

    def add_imgs(self, preds, targets):
        pass

    def get_default_metric(self):
        return 0

    def reset(self):
        pass

    def get_all(self):
        return 0

    def print_all(self):
        print(self.get_all())


class ObjectDetectionMetrics(object):
    def __init__(self, default_metric='map50'):
        self.metric = MeanAveragePrecision(class_metrics=False)
        self.metric.warn_on_many_detections = False
        self.default_metric = default_metric
        self.preds = []
        self.targets = []

    def add_imgs(self, preds, targets):
        self.preds = [{k: v.detach() for k, v in p.items()} for p in preds]
        self.targets = [{k: v.detach() for k, v in t.items()} for t in targets]
        self.metric.update(self.preds, self.targets)

    def get_default_metric(self):
        if self.default_metric =='map50':
            return self.mAP50()
        elif self.default_metric =='map75':
            return self.mAP75()
        elif self.default_metric =='map':
            return self.mAP()
    
    def reset(self):
        self.preds = []
        self.targets = []
        self.metric.reset()

    def mAP50(self):
        return self.metric.compute()['map_50'].item()
    
    def mAP75(self):
        return self.metric.compute()['map_75'].item()

    def mAP(self):
        ret = self.metric.compute()['map'].item()
        return ret
    
    def get_all(self):
        return self.metric.compute()
    
    def print_all(self):
        print(self.get_all())


class ImageClassificationMetrics(object):
    def __init__(self, num_classes, default_metric='top1acc', ignore_index=None):
        self.num_classes = num_classes
        self.default_metric = default_metric
        self.ignore_index = ignore_index
        if ignore_index is not None:
            assert (0 <= ignore_index < num_classes), 'ignore_index应在class范围内'
        self.logs = ''

        if self.num_classes == 2:
            task = 'binary'
        elif self.num_classes > 2:
            task ='multiclass'
        else:
            raise ValueError('num_classes must be greater than 1')
        
        self.top1acc = torchmetrics.Accuracy(task=task, num_classes=self.num_classes, average='none', ignore_index=self.ignore_index, sync_on_compute=False, top_k=1).cuda()
        self.top5acc = torchmetrics.Accuracy(task=task, num_classes=self.num_classes, average='none', ignore_index=self.ignore_index, sync_on_compute=False, top_k=5).cuda()
        self.micro_top1acc = torchmetrics.Accuracy(task=task, num_classes=self.num_classes, average='micro', ignore_index=self.ignore_index, sync_on_compute=False, top_k=1).cuda()
        self.micro_top5acc = torchmetrics.Accuracy(task=task, num_classes=self.num_classes, average='micro', ignore_index=self.ignore_index, sync_on_compute=False, top_k=5).cuda()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task=task, num_classes=self.num_classes, ignore_index=self.ignore_index, sync_on_compute=False).cuda()

    def add_imgs(self, preds, labels):
        self.top1acc.update(preds, labels)
        self.top5acc.update(preds, labels)
        self.micro_top1acc.update(preds, labels)
        self.micro_top5acc.update(preds, labels)
        self.confusion_matrix.update(preds, labels)

    def get_default_metric(self):
        if self.default_metric =='top1acc':
            return self.micro_top1acc.compute().item()
        elif self.default_metric =='top5acc':
            return self.micro_top5acc.compute().item()
        else:
            raise ValueError('default_metric must be top1acc or top5acc')

    def reset(self):
        self.top1acc.reset()
        self.top5acc.reset()
        self.micro_top1acc.reset()
        self.micro_top5acc.reset()
        self.confusion_matrix.reset()

    
    def get_all(self):
        log = f'class_Top1Acc: {self.top1acc.compute()}\n'
        log += f'class_Top5Acc: {self.top5acc.compute()}\n'
        log += f'Micro_Top1Acc: {self.micro_top1acc.compute()}\n'
        log += f'Micro_Top5Acc: {self.micro_top5acc.compute()}\n'
        log += f'Confusion_Matrix: {self.confusion_matrix.compute()}\n'
        return log
    
    def print_all(self):
        print(self.get_all())


class SemanticSegmentationMetrics(object):
    def __init__(self, num_classes, class_list, default_metric='mf1', ignore_index=None):
        self.class_list = class_list
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if ignore_index is not None:
            assert (0 <= ignore_index < num_classes), 'ignore_index应在class范围内'
        self.default_metric = default_metric
        self.logs = ''
        
        self.acc = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, average='micro', ignore_index=self.ignore_index, sync_on_compute=False).cuda()
        self.prec = torchmetrics.classification.MulticlassPrecision(num_classes=self.num_classes, average='none', ignore_index=self.ignore_index, sync_on_compute=False).cuda()
        self.rec = torchmetrics.classification.MulticlassRecall(num_classes=self.num_classes, average='none', ignore_index=self.ignore_index, sync_on_compute=False).cuda()
        self.F1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='none', ignore_index=self.ignore_index, sync_on_compute=False).cuda()
        self.IoU = torchmetrics.classification.MulticlassJaccardIndex(num_classes=self.num_classes, average='none', ignore_index=self.ignore_index, sync_on_compute=False).cuda()
        self.confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=self.num_classes, ignore_index=self.ignore_index, sync_on_compute=False).cuda()
        

    def add_imgs(self, preds, labels):
        '''
        preds: (batch, num_classes, Height, widths) torch.float32
        labels: (batch, num_classes, Height, width) torch.uint8
        '''
        preds = preds.argmax(dim=1).type(torch.uint8)
        labels = labels.argmax(dim=1).type(torch.uint8)

        self.acc.update(preds, labels)
        self.prec.update(preds, labels)
        self.rec.update(preds, labels)
        self.F1.update(preds, labels)
        self.IoU.update(preds, labels)
        self.confusion_matrix.update(preds, labels)


    def get_default_metric(self):
        if self.default_metric =='miou':
            return self.macro_IoU()
        elif self.default_metric == 'mf1':
            return self.macro_F1score()
        else:
            raise ValueError('default_metric must be mIoU or mF1')

    def reset(self):
        self.acc.reset()
        self.prec.reset()
        self.rec.reset()
        self.F1.reset()
        self.IoU.reset()
        self.confusion_matrix.reset()

    def macro_precision(self):
        class_prec = self.prec.compute()
        div = self.num_classes if self.ignore_index is None else self.num_classes - 1
        return (class_prec.sum() / div).item()

    def macro_recall(self):
        class_rec = self.rec.compute()
        div = self.num_classes if self.ignore_index is None else self.num_classes - 1
        return (class_rec.sum() / div).item()

    def macro_IoU(self):
        class_iou = self.IoU.compute()
        div = self.num_classes if self.ignore_index is None else self.num_classes - 1
        return (class_iou.sum() / div).item()

    def macro_F1score(self):
        class_F1 = self.F1.compute()           
        div = self.num_classes if self.ignore_index is None else self.num_classes - 1
        return (class_F1.sum() / div).item()
    
    def get_all(self):
        log = f'PA: {self.acc.compute()}\n'
        log += f'class_precision: {self.prec.compute()}\n'
        log += f'class_recall: {self.rec.compute()}\n'
        log += f'class_F1: {self.F1.compute()}\n'
        log += f'class_IoU: {self.IoU.compute()}\n'
        log += f'confusion_matrix: {self.confusion_matrix.compute()}\n'
        log += f'macro_precision: {self.macro_precision()}\n'
        log += f'macro_recall: {self.macro_recall()}\n'
        log += f'macro_IoU: {self.macro_IoU()}\n'
        log += f'macro_F1score: {self.macro_F1score()}\n'
        return log
    
    def print_all(self):
        print(self.get_all())



class ChangeDetectionMetrics(object):
    def __init__(self, num_classes, class_list, mask_id):
        self.class_list = class_list
        self.num_classes = num_classes
        self.mask_id = mask_id
        self.metricsA = SemanticSegmentationMetrics(self.num_classes, self.class_list)
        self.metricsB = SemanticSegmentationMetrics(self.num_classes, self.class_list)
        self.metricsmask = SemanticSegmentationMetrics(self.num_classes, [0, 1])

    def add_imgs(self, preds, labels):
        predsA, predsB, predsmask = preds
        labelsA, labelsB, masks = labels

        predsA = np.array(predsA.argmax(dim=1).type(torch.uint8).to('cpu'))
        predsB = np.array(predsB.argmax(dim=1).type(torch.uint8).to('cpu'))
        predsmask = np.array(predsmask.argmax(dim=1).type(torch.uint8).to('cpu'))

        # labelsA = np.array(batch_one_hot_decode(labelsA, list(range(len(self.class_list)))).to('cpu'))
        # labelsB = np.array(batch_one_hot_decode(labelsB, list(range(len(self.class_list)))).to('cpu'))
        # masks = np.array(batch_one_hot_decode(masks, [0, 1]).to('cpu'))

        self.metricsA.add_imgs(predsA, labelsA)
        self.metricsB.add_imgs(predsB, labelsB)
        self.metricsmask.add_imgs(predsmask, masks)

    def reset(self):
        self.metricsA.reset()
        self.metricsB.reset()
        self.metricsmask.reset()

    def get_default_metric(self):
        return (self.metricsA.get_default_metric(), self.metricsB.get_default_metric(), self.metricsmask.get_default_metric()) / 3
    
    def pixel_accuracy(self, get_all=False):
        if get_all:
            return (self.metricsA.pixel_accuracy(), self.metricsB.pixel_accuracy(), self.metricsmask.pixel_accuracy())
        else:
            return (self.metricsA.pixel_accuracy() + self.metricsB.pixel_accuracy() + self.metricsmask.pixel_accuracy()) / 3
        
    def macro_F1score(self, get_all=False):
        if get_all:
            return (self.metricsA.macro_F1score(), self.metricsB.macro_F1score(), self.metricsmask.macro_F1score())
        else:
            return (self.metricsA.macro_F1score() + self.metricsB.macro_F1score() + self.metricsmask.macro_F1score()) / 3
        
    def macro_IoU(self, get_all=False):
        if get_all:
            return (self.metricsA.macro_IoU(), self.metricsB.macro_IoU(), self.metricsmask.macro_IoU())
        else:
            return (self.metricsA.macro_IoU() + self.metricsB.macro_IoU() + self.metricsmask.macro_IoU()) / 3
        
    def macro_precision(self, get_all=False):
        if get_all:
            return (self.metricsA.macro_precision(), self.metricsB.macro_precision(), self.metricsmask.macro_precision())
        else:
            return (self.metricsA.macro_precision() + self.metricsB.macro_precision() + self.metricsmask.macro_precision()) / 3
        
    def macro_recall(self, get_all=False):
        if get_all:
            return (self.metricsA.macro_recall(), self.metricsB.macro_recall(), self.metricsmask.macro_recall())
        else:
            return (self.metricsA.macro_recall() + self.metricsB.macro_recall() + self.metricsmask.macro_recall()) / 3
        
    def get_all(self):
        return (self.metricsA.get_all(), self.metricsB.get_all(), self.metricsmask.get_all())
    
    def print_all(self):
        print(self.get_all())


def metric_test():
    preds = torch.tensor([[0, 1, 0],
                           [2, 1, 0],
                           [2, 2, 1]])
    labels = torch.tensor([[1, 1, 1],
                           [2, 1, 1],
                           [2, 2, 1]])
    metrics = SemanticSegmentationMetrics(3)
    metrics.add_imgs(preds, labels)
    metrics.print_all()


if __name__ == '__main__':
    metric_test()
