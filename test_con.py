import torch.nn as nn
import torch.tensor as tensor
import torch
import f
# import numpy as np
# loss=nn.CrossEntropyLoss()
# a=tensor(([2,3],[4,5]),dtype=torch.float)
# w=tensor(torch.ones(2,1),dtype=torch.float,requires_grad=True)
# out=torch.mm(a,w)
# print(out)
# a=5
# print(tensor(5,dtype=torch.float))
# out=torch.mul(out.float(),tensor(5).float())
# # print(out)
# # print(tensor([1]).float)
# sm=nn.Softmax(dim=0)
# print(out.view(-1))
# smo=sm(out.view(-1))
# print(smo)
# smo=torch.log(smo)
# loss=nn.NLLLoss()
# target=tensor([1])
# loss=loss(smo.unsqueeze(0),target)
# print(loss)
# loss.backward()
# print(w.grad.data)

def test_grad():
    input=tensor(([1,2,3],[4,5,6],[7,8,9]),dtype=torch.float)
    #weight=tensor(([0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4]),requires_grad=True)
    weight=tensor(torch.rand(3, 4),requires_grad=True)
    #input=input.unsqueeze(0)
    print(input,weight)
    pre=torch.mm(input,weight)
    #loss1=f.multilabel_soft_margin_loss()
    loss2=nn.MultiLabelMarginLoss()
    lable1=tensor(([0, 1, 1,0],),dtype=torch.float)
    lable2 = tensor(([0, 1, 1,0], [1, 0, 0,0], [1, 0,1 ,1]), dtype=torch.long)
    print(pre,lable1)
    loss1=f.multilabel_soft_margin_loss(pre,lable1,reduction='sum')
    loss1.backward()
    print('weight.grad.data1:',weight.grad.data)

    # loss2 = loss2(pre, lable2)
    # loss2.backward()
    # print('weight.grad.data2:', weight.grad.data)
if __name__ == '__main__':
    test_grad()