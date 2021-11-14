import models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np

from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import SubsetRandomSampler
from Data.utils import *
from Data.cifar import CIFAR10, CIFAR100
from train_valid import *
from parser_args import *

def main_worker_pbar(args):
    global best_acc1
    softmax = nn.Softmax(dim = 1)
    pth = ''

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = models.__dict__[args.arch]()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True


    if args.dataset=='cifar10':
        args.top_bn = False
        p = torch.empty((args.K,args.N,50000,10))
        args.epoch_decay_start = 80
        args.n_epoch = 200
        train_dataset = CIFAR10(root='./Data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )
                            
    
        test_dataset = CIFAR10(root='./Data/',
                                    download=True,  
                                    train=False, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )

    if args.dataset=='cifar100':
        args.top_bn = False
        args.epoch_decay_start = 100
        p = torch.empty((args.K,args.N,50000,100))
        args.n_epoch = 200
        train_dataset = CIFAR100(root='./Data/',
                                    download=True,  
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )
    
        test_dataset = CIFAR100(root='./Data/',
                                    download=True,  
                                    train=False, 
                                    transform=transforms.ToTensor(),
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                )
    if args.noise_type !='clean' and args.noise_type !='saltpepper':
        noise_or_not = train_dataset.noise_or_not
        torch.save(noise_or_not,pth + args.dataset+'_'+str(args.noise_rate)+'/noise.pt')
        print("total length:",noise_or_not.shape[0],"noise:",noise_or_not.shape[0]-np.sum(noise_or_not))
    if args.noise_type =='saltpepper':
        salt_or_not = train_dataset.salt_or_not
        torch.save(salt_or_not,pth + args.dataset+'_'+str(args.noise_rate)+'/salt.pt')
        print("total length:",salt_or_not.shape[0],"sault:",np.sum(salt_or_not))


    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=False)
    
    testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train_pbar.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test_pbar.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for k in range(args.K):
        for n in range(args.N):
            split1 = n/args.N
            split2 = (n+1)/args.N
            dataset_size = len(train_dataset)
            indices = list(range(dataset_size))
            split_place1 = int(np.floor(split1 * dataset_size))
            split_place2 = int(np.floor(split2 * dataset_size))-1
            mini_train_indices = indices[:split_place1]+indices[split_place2:]
            mini_train_sampler = SubsetRandomSampler(mini_train_indices)
            mini_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,sampler=mini_train_sampler)
            
            best_acc1 = 0
            optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6, momentum=args.momentum, weight_decay=args.weight_decay) 
            checkpoint = torch.load(pth + '/'+args.dataset+'_'+args.noise_type+'_'+str(args.noise_rate)+'_'+args.arch +'/ckpt.best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'],False)
            for epoch in range(args.start_epoch, 100):
                criterion = nn.CrossEntropyLoss().cuda(args.gpu)


                # train for one epoch
                train(mini_trainloader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        
                # evaluate on validation set
                acc1 = validate(testloader, model, criterion, epoch, args, log_testing, tf_writer)

                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)

                tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
                output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
                output_acc1 = 'Prec@1: %.3f' % acc1
                print("epoch:",epoch+1,output_acc1,output_best)
                log_testing.write(output_best + '\n')
                log_testing.flush()

                save_checkpoint_pbar(args, {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)

            

            checkpoint = torch.load(pth + '/'+args.dataset+'_'+args.noise_type+'_'+str(args.noise_rate)+'_'+args.arch +'/ckpt.best.pbar.pth.tar')
            model.load_state_dict(checkpoint['state_dict'],False)
            model.eval()
            with torch.no_grad():
                for i,data in enumerate(trainloader, 0):
                    input, target, indexes = data
                    ind=indexes.cpu().numpy().transpose()

                    if args.gpu is not None:
                        input = input.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True).long()

                    # compute output
                    output = model(input)
                    Out_soft = softmax(output)
                    for j in range(target.size(0)):
                        p[k,n,128*i+j] = Out_soft[j]
                
            print("k:",k+1,"n:",n+1,"best accuracy:",best_acc1)
            #print(p[k,n,25000])
                
    #torch.save(p,pth+'/P.pt')
    Pbar = trainpbar(p,args.K,args.N)
    Var,Bias = trainErr(trainloader,p,Pbar,args)
    torch.save(Var,pth+'/Variance.pt')
    torch.save(Bias,pth+'/Bias.pt')

    Lam = np.array([0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4])
    T = np.array([0.8,0.9,1,1.1,1.2])
    result = np.zeros((5,10))
    recall = np.zeros((5,10))
    A = np.zeros((5,10))
    for l in range(10):
        L = Lam[l]
        Err = Bias + L*Var
        for t in range(5):
            noise = int(50000*args.noise_rate*T[t])
            split = 50000-noise
            index = np.argsort(Err)
            index = index[split:]
            a = 0
            for i in range(index.shape[0]):
                if salt_or_not[index[i]]:
                        a += 1
            A[t,l] = a
            result[t,l] = format(a/float(noise),'.4f')
            recall[t,l] = format(a/float(np.sum(salt_or_not)),'.4f')
    torch.save(A,pth+'/A.pt')
    print(args.noise_rate)
    print(A)
    print("result:\n",result)
    print("recall:\n",recall)
    torch.save(result,pth+'/result.pt')
    torch.save(recall,pth+'/recall.pt')