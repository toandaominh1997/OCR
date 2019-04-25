import torch
def by_field(predict, target):
    with torch.no_grad():
        correct=0
        targets = []
        for i in target:
            targets.append(i.decode('utf-8', 'strict'))
        for pred, tar in zip(predict, targets):
            if(pred==tar):
                correct+=1
    return correct / float(len(targets))

def by_char(predict, target):
    with torch.no_grad():
        targets = []
        correct = 0
        total_target = 0
        for i in target:
            targets.append(i.decode('utf-8', 'strict'))
        for pred, tar in zip(predict, targets):
            total_target+=len(tar)
            for p, t in zip(pred, tar):
                if(p==t):
                    correct+=1
    return correct/float(total_target)