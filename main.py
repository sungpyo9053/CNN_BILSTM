import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse

from dataloader import data_loader


from evaluation import evaluation_metrics
from CNN_BiLSTM_model import Net
from BI_LSTM import Net
#from TFIDF_model import tfidfNet


try:
    from nipa import nipa_data
    DATASET_PATH = nipa_data.get_data_root('deepfake')
except:
    DATASET_PATH = os.path.join('./data')


def _infer(model, cuda, data_loader):
    res_comment = []
    res_classes = []
    for index, data in enumerate(data_loader):
        (comment, comment_vec), label = data
        if cuda:
            comment_vec = comment_vec.cuda()
        
        pred = model(comment_vec)
        pred = pred.detach().cpu().numpy()
        res_classes.append(pred[0])
        res_comment.append(comment)
        
    res_classes = np.argmax(res_classes, axis=1)
    return [res_comment, res_classes]


def feed_infer(output_file, infer_func):
    prediction_comment, prediction_class = infer_func()
    bias_name_list = ['none', 'gender', 'others']
    hate_name_list = ['none', 'hate', 'offensive']
    from itertools import product
    bias_hate_list = [bias_name_list, hate_name_list]
    bias_hate_list = list(product(*bias_hate_list))
    
    print('write output')
    predictions_str = []
    for index, name in enumerate(prediction_comment):
        bias = bias_hate_list[prediction_class[index]][0]
        hate = bias_hate_list[prediction_class[index]][1]
        #test_str = name[0] + '\t' + bias + '\t' + hate
        test_str = bias + '\t' + hate
        predictions_str.append(test_str)
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def validate(prediction_file, model, validate_dataloader, validate_label_file, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=validate_dataloader))
    metric_result = evaluation_metrics(prediction_file, validate_label_file)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result


def test(prediction_file, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader))


def save_model(model_name, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(DATASET_PATH, model_name + '.pth'))
    print('model saved')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(DATASET_PATH, model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')


if __name__ == '__main__':
    
    

    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=9)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=30)
    args.add_argument("--model_name", type=str, default="model.pth")
    args.add_argument("--prediction_file", type=str, default="prediction.txt")
    args.add_argument("--batch", type=int, default=30) # 
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--momentum", type=float, default =0.5)
    args.add_argument("--model", type=str, default ="CNN_BILSTM")
    
    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode
    model = config.model
    momentums= config.momentum
    validate_label_file = './data/validate_hate.txt'
    
    
        
    hidden_size = 64
    num_layer = 1
    embeding_dim = 128
    n_vocab = 6581
    max_length= 74
    
    
    if model == "CNN_BILSTM":
        model = Net(hidden_size,num_layer,num_classes,embeding_dim,n_vocab,max_length,drop_prob=0.5)
        print(model)
    
    if model == "BI_LSTM":
        model = Net(hidden_size,num_layer,num_classes,embeding_dim,n_vocab,max_length,drop_prob=0.5)
        print(model)

    if mode == 'test':
        load_model(model_name, model)

    if cuda:
        model = model.cuda()
        
    if mode == 'train':
        # define loss function
        loss_fn = nn.CrossEntropyLoss()
        if cuda:
            loss_fn = loss_fn.cuda()

        # set optimizer #Adam
        optimizer = Adam([param for param in model.parameters() if param.requires_grad],lr=base_lr, weight_decay=1e-4)
        #optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad], lr=base_lr, momentum=0.5)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

        # get data loader
        train_dataloader = data_loader(root=DATASET_PATH, phase='train', batch_size=batch)
        validate_dataloader = data_loader(root=DATASET_PATH, phase='validate', batch_size=1)
        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader)

        
        #check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter : ",total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :",trainable_params)
        print("------------------------------------------------------------")
        global padding_length
        # train
        for epoch in range(num_epochs):
            model.train()
            for iter_, data in enumerate(train_dataloader):
                #print(len(train_dataloader))
                
                (_, comment_vec), label = data
                
                #padding_length = torch.LongTensor([torch.max(comment_vec[i, :].data.nonzero())+1 for i in range(comment_vec.size(0))])
                #padding_length, sorted_idx = padding_length.sort(0, descending=True)

                if cuda:
                    comment_vec = comment_vec.cuda()
            
                label = label.type(torch.LongTensor)
                pred = model(comment_vec)
                label = label.cuda()
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) '
                          'elapsed {} expected per epoch {}'.format(
                              _epoch, num_epochs, loss.item(), elapsed, expected))
                    time_ = datetime.datetime.now()
            
            # scheduler update
            scheduler.step()

            # validate
            validate(prediction_file, model, validate_dataloader, validate_label_file, cuda)

            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))
        save_model(str(epoch + 1), model, optimizer, scheduler)
        
    elif mode == 'test':
        model.eval()
        # get data loader
        test_dataloader = data_loader(root=DATASET_PATH, phase='test', batch_size=batch)
        test(prediction_file, model, test_dataloader, cuda)
        # submit test result

