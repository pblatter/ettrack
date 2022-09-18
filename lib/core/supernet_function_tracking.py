import math
import time
import torch
import numpy as np
#from lib.utils.utils import print_speed
import random
import matplotlib.pyplot as plt


def get_cand_with_prob(CHOICE_NUM, prob=None, sta_num=(4,4,4,4,4)):
    if prob is None:
        get_random_cand = [np.random.choice(CHOICE_NUM, item).tolist() for item in sta_num]
    else:
        get_random_cand = [np.random.choice(CHOICE_NUM, item, prob).tolist() for item in sta_num]
    # print(get_random_cand)
    return get_random_cand

def get_cand_head():
    oup = [random.randint(0, 2)] # num of channels (3 choices)
    arch = []
    arch.append(random.randint(0, 1)) # 3x3 conv, 5x5 conv
    arch += list(np.random.choice(3,7)) # 3x3 conv, 5x5 conv, skip
    oup.append(arch)
    return oup

'''2020.10.24 Without using IDentity'''
def get_cand_head_wo_ID():
    oup = [random.randint(0, 2)] # num of channels (3 choices)
    arch = []
    arch.append(random.randint(0, 1)) # 3x3 conv, 5x5 conv
    arch += list(np.random.choice(2,7)) # 3x3 conv, 5x5 conv
    oup.append(arch)
    return oup


'''2020.10.15 get a random output position'''
# def get_oup_pos(sta_num):
#     stage_idx = random.randint(1,3) # 1,2,3
#     block_num = sta_num[stage_idx]
#     block_idx = random.randint(0, block_num-1)
#     return [stage_idx, block_idx]
'''2020.10.29 we don't sample feature with stride=8'''
def get_oup_pos(sta_num):
    stage_idx = random.randint(2,3) # 1,2,3
    block_num = sta_num[stage_idx]
    block_idx = random.randint(0, block_num-1)
    return [stage_idx, block_idx]

'''2020.10.15 get a random output position'''
def get_oup_pos_effib7():
    return random.randint(11,37)

'''2020.10.5 name --> path'''
'''2020.10.17 modified version'''
def name2path_backhead(path_name, sta_num=(4,4,4,4,4), head_only=False, backbone_only=False):
    backbone_name, head_name = path_name.split('+cls_')
    if not head_only:
        # process backbone
        backbone_name = backbone_name.strip('back_')[1:-1] # length = 20 when 600M, length = 18 when 470M
        backbone_path = [[],[],[],[],[]]
        for stage_idx in range(len(sta_num)):
            for block_idx in range(sta_num[stage_idx]):
                str_idx = block_idx + sum(sta_num[:stage_idx])
                backbone_path[stage_idx].append(int(backbone_name[str_idx]))
        backbone_path.insert(0, [0])
        backbone_path.append([0])
    if not backbone_only:
        # process head
        cls_name, reg_name = head_name.split('+reg_')
        head_path = {}
        cls_path = [int(cls_name[0])]
        cls_path.append([int(item) for item in cls_name[1:]])
        head_path['cls'] = cls_path
        reg_path = [int(reg_name[0])]
        reg_path.append([int(item) for item in reg_name[1:]])
        head_path['reg'] = reg_path
    # combine
    if head_only:
        backbone_path = None
    if backbone_only:
        head_path = None
    return tuple([backbone_path, head_path])


'''2020.10.5 name --> path'''
'''2020.10.17 modified version'''
def name2path(path_name, sta_num=(4,4,4,4,4), head_only=False, backbone_only=False):
    if '_ops_' in path_name:
        first_name, ops_name = path_name.split('_ops_')
        backbone_path, head_path = name2path_backhead(first_name, sta_num=sta_num, head_only=head_only, backbone_only=backbone_only)
        ops_path = (int(ops_name[0]),int(ops_name[1]))
        return (backbone_path, head_path, ops_path)
    else:
        return name2path_backhead(path_name, sta_num=sta_num, head_only=head_only, backbone_only=backbone_only)

def name2path_ablation(path_name, sta_num=(4,4,4,4,4), num_tower=8):
    back_path, head_path, ops_path = None, None, None
    if 'back' in path_name:
        back_str_len = sum(sta_num) + 2 # head0, tail0
        back_str = path_name.split('back_')[1][:back_str_len]
        back_str = back_str[1:-1] # remove head0 and tail0
        back_path = [[],[],[],[],[]]
        for stage_idx in range(len(sta_num)):
            for block_idx in range(sta_num[stage_idx]):
                str_idx = block_idx + sum(sta_num[:stage_idx])
                back_path[stage_idx].append(int(back_str[str_idx]))
        back_path.insert(0, [0])
        back_path.append([0])
    if 'cls' in path_name and 'reg' in path_name:
        head_path = {}
        cls_str_len = num_tower + 1 # channel idx
        cls_str = path_name.split('cls_')[1][:cls_str_len]
        cls_path = [int(cls_str[0]), [int(item) for item in cls_str[1:]]]
        head_path['cls'] = cls_path
        reg_str_len = num_tower + 1 # channel idx
        reg_str = path_name.split('reg_')[1][:reg_str_len]
        reg_path = [int(reg_str[0]), [int(item) for item in reg_str[1:]]]
        head_path['reg'] = reg_path
    if 'ops' in path_name:
        ops_str = path_name.split('ops_')[1]
        ops_path = (int(ops_str[0]), int(ops_str[1]))
    return {'back':back_path, 'head': head_path, 'ops': ops_path}


if __name__ == "__main__":
    for _ in range(10):
        print(get_cand_head())
# -----------------------------
# Main training code for Ocean
# -----------------------------
'''Randomly sample paths for training'''
def supernet_train(train_loader, model, optimizer, epoch, cur_lr, cfg,
                   writer_dict, logger, sta_num, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)
    '''Get some information about the supernet'''
    CHOICE_NUM = 6

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)

        '''2020.09.07 Sample a path randomly'''
        # backbone
        get_random_cand = get_cand_with_prob(CHOICE_NUM, sta_num=sta_num)
        # add head and tail
        get_random_cand.insert(0, [0])
        get_random_cand.append([0])
        # head
        cand_h_dict = {}
        cand_h_dict['cls'] = get_cand_head()
        cand_h_dict['reg'] = get_cand_head()

        cls_loss_ori, reg_loss = model(template, search, label_cls, reg_label, reg_weight,
                                       get_random_cand, cand_h_dict)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)


        cls_loss_align = 0
        loss = cls_loss_ori +  1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item() # 当前结果
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

'''2020.10.05 Retrain the supernet (The path is specified)'''
def supernet_retrain(train_loader, model, optimizer, epoch, cur_lr, cfg,
                   writer_dict, logger, device):

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)

        cls_loss_ori, reg_loss = model(template, search, label_cls, reg_label, reg_weight)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        loss = cls_loss_ori + 1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

'''Randomly sample paths for training'''
'''2020.10.15 Introducing random output position'''
def supernet_train_DP(train_loader, model, optimizer, epoch, cur_lr, cfg,
                   writer_dict, logger, sta_num, device):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)
    '''Get some information about the supernet'''
    CHOICE_NUM = 6

    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        template, search = input[0].to(device), input[1].to(device)
        '''stride=16'''
        label_cls_16 = input[2].type(torch.FloatTensor).to(device)  # BCE need float
        reg_label_16, reg_weight_16 = input[3].float().to(device), input[4].float().to(device)
        '''stride=8'''
        label_cls_8 = input[5].type(torch.FloatTensor).to(device)  # BCE need float
        reg_label_8, reg_weight_8 = input[6].float().to(device), input[7].float().to(device)

        '''2020.09.07 Sample a path randomly'''
        # backbone
        get_random_cand = get_cand_with_prob(CHOICE_NUM, sta_num=sta_num)
        # add head and tail
        get_random_cand.insert(0, [0])
        get_random_cand.append([0])
        # head
        cand_h_dict = {}
        cand_h_dict['cls'] = get_cand_head()
        cand_h_dict['reg'] = get_cand_head()
        # backbone output position
        oup_pos = get_oup_pos(sta_num)
        # print('oup_pos is ',oup_pos)
        stage_idx = oup_pos[0]
        if stage_idx == 1:
            # print('using stride 8')
            cls_loss_ori, reg_loss = model(template, search, label_cls_8, reg_label_8, reg_weight_8,
                                       get_random_cand, cand_h_dict, oup_pos)
        elif stage_idx == 2 or stage_idx == 3:
            # print('using stride 16')
            cls_loss_ori, reg_loss = model(template, search, label_cls_16, reg_label_16, reg_weight_16,
                                           get_random_cand, cand_h_dict, oup_pos)
        else:
            raise ValueError ('stage_idx should be 1,2,3')
        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)


        cls_loss_align = 0
        loss = cls_loss_ori + 1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict

'''Randomly sample paths for training'''
'''2020.10.15 Introducing random output position'''
'''2020.10.22 For ablation study'''
def supernet_train_DP_ablation(train_loader, model, optimizer, epoch, cur_lr, cfg,
                   writer_dict, logger, search_dict, sta_num=None, device='cuda'):
    # unfix for FREEZE-OUT method
    # model, optimizer = unfix_more(model, optimizer, epoch, cfg, cur_lr, logger)

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)

    '''Get some information about the supernet'''
    CHOICE_NUM = 6
    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        template, search = input[0].to(device), input[1].to(device)
        '''stride=16'''
        label_cls_16 = input[2].type(torch.FloatTensor).to(device)  # BCE need float
        reg_label_16, reg_weight_16 = input[3].float().to(device), input[4].float().to(device)
        if search_dict['search_out']:
            '''stride=8'''
            label_cls_8 = input[5].type(torch.FloatTensor).to(device)  # BCE need float
            reg_label_8, reg_weight_8 = input[6].float().to(device), input[7].float().to(device)

        '''2020.09.07 Sample a path randomly'''
        # backbone
        if search_dict['search_back']:
            get_random_cand = get_cand_with_prob(CHOICE_NUM, sta_num=sta_num)
            # add head and tail
            get_random_cand.insert(0, [0])
            get_random_cand.append([0])
        else:
            get_random_cand = None
        # head
        if search_dict['search_head']:
            cand_h_dict = {}
            cand_h_dict['cls'] = get_cand_head()
            cand_h_dict['reg'] = get_cand_head()
        else:
            cand_h_dict = {'cls':[0,[0,0,0,0,0,0,0,0]], 'reg':[0,[0,0,0,0,0,0,0,0]]}
        # backbone output position
        if search_dict['search_out']:
            oup_pos = get_oup_pos(sta_num)
            # print('oup_pos is ',oup_pos)
            stage_idx = oup_pos[0]
            if stage_idx == 1:
                # print('using stride 8')
                cls_loss_ori, reg_loss = model(template, search, label_cls_8, reg_label_8, reg_weight_8,
                                               get_random_cand, cand_h_dict, oup_pos)
            elif stage_idx == 2 or stage_idx == 3:
                # print('using stride 16')
                cls_loss_ori, reg_loss = model(template, search, label_cls_16, reg_label_16, reg_weight_16,
                                               get_random_cand, cand_h_dict, oup_pos)
            else:
                raise ValueError('stage_idx should be 1,2,3')
        else:
            oup_pos = None
            cls_loss_ori, reg_loss = model(template, search, label_cls_16, reg_label_16, reg_weight_16,
                                           get_random_cand, cand_h_dict, oup_pos)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        cls_loss_align = 0
        loss = cls_loss_ori + 1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict




def train_DP_baseline(loaders, model, optimizer, epoch, cur_lr, cfg,
                   writer_dict, logger, device='cuda'):

    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)

    for l in loaders.keys():

        # prepare
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        cls_losses_align = AverageMeter()
        cls_losses_ori = AverageMeter()
        reg_losses = AverageMeter()
        end = time.time()

        loader = loaders[l]
    
        for iter, input in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # input and output/loss
            template, search = input[0].to(device), input[1].to(device)
            '''stride=16'''
            label_cls_16 = input[2].type(torch.FloatTensor).to(device)  # BCE need float
            reg_label_16, reg_weight_16 = input[3].float().to(device), input[4].float().to(device)
            
            if l=='train':
                model.train()
            elif l=='val':
                model.eval()
            else:
                raise NameError('not real mode')

            cls_loss_ori, reg_loss = model(template, search, label_cls_16, reg_label_16, reg_weight_16)  

            cls_loss_ori = torch.mean(cls_loss_ori)
            reg_loss = torch.mean(reg_loss)

            cls_loss_align = 0
            loss = cls_loss_ori + 1.2 * reg_loss

            loss = torch.mean(loss)

            # if training -> compute gradient and do update step
            if l=='train':
                optimizer.zero_grad()
                loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

            if is_valid_number(loss.item()):
                optimizer.step()

            # record loss
            loss = loss.item()
            losses.update(loss, template.size(0))

            cls_loss_ori = cls_loss_ori.item()
            cls_losses_ori.update(cls_loss_ori, template.size(0))

            try:
                cls_loss_align = cls_loss_align.item()
            except:
                cls_loss_align = 0

            cls_losses_align.update(cls_loss_align, template.size(0))

            reg_loss = reg_loss.item()
            reg_losses.update(reg_loss, template.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (iter + 1) % cfg.PRINT_FREQ == 0:
                logger.info(
                    'Mode: {l}\t Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                        epoch, iter + 1, len(loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                        loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses, l=l))

                print_speed((epoch - 1) * len(loader) + iter + 1, batch_time.avg,
                            cfg.OCEAN.TRAIN.END_EPOCH * len(loader), logger)

            # write to tensorboard

            if l=='train':

                # training
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', loss, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            elif l=='val':
                
                # validation
                writer = writer_dict['writer']
                val_global_steps = writer_dict['val_global_steps']
                train_global_steps = writer_dict['train_global_steps']
                writer.add_scalar('val_loss_single', loss, val_global_steps)
                writer.add_scalar('val_loss', loss, train_global_steps)
                writer_dict['val_global_steps'] = global_steps + 1



    return model, writer_dict


'''2020.10.29 General Training pipeline'''
def supernet_train_DP_general(train_loader, model, optimizer, epoch, cur_lr, cfg,
                   writer_dict, logger, search_dict, toolbox, device='cuda'):

    # prepare
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_align = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    '''NOT HERE TO USE MODEL.TRAIN()'''
    model = model.to(device)

    '''Get some information about the supernet'''
    CHOICE_NUM = 6
    for iter, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input and output/loss
        template, search = input[0].to(device), input[1].to(device)
        '''stride=16'''
        label_cls_16 = input[2].type(torch.FloatTensor).to(device)  # BCE need float
        reg_label_16, reg_weight_16 = input[3].float().to(device), input[4].float().to(device)

        '''2020.09.07 Sample a path randomly'''
        cand_dict = toolbox.get_one_path()
        get_random_cand, cand_h_dict, oup_pos = cand_dict['back'], cand_dict['head'], cand_dict['ops']
        print("cand h dict: ", cand_h_dict)

        # backbone output position
        if search_dict['search_out']:
            # print('oup_pos is ',oup_pos)
            stage_idx = oup_pos[0]
            if stage_idx == 1:
                # print('using stride 8')
                '''stride=8'''
                label_cls_8 = input[5].type(torch.FloatTensor).to(device)  # BCE need float
                reg_label_8, reg_weight_8 = input[6].float().to(device), input[7].float().to(device)
                cls_loss_ori, reg_loss = model(template, search, label_cls_8, reg_label_8, reg_weight_8,
                                               get_random_cand, cand_h_dict, oup_pos)
            elif stage_idx == 2 or stage_idx == 3:
                # print('using stride 16')
                cls_loss_ori, reg_loss = model(template, search, label_cls_16, reg_label_16, reg_weight_16,
                                               get_random_cand, cand_h_dict, oup_pos)
            else:
                raise ValueError('stage_idx should be 1,2,3')
        else:
            cls_loss_ori, reg_loss = model(template, search, label_cls_16, reg_label_16, reg_weight_16,
                                           get_random_cand, cand_h_dict, oup_pos)

        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        cls_loss_align = 0
        loss = cls_loss_ori + 1.2 * reg_loss

        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_align = cls_loss_align.item()
        except:
            cls_loss_align = 0

        cls_losses_align.update(cls_loss_align, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_ALIGN Loss:{cls_loss_align.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_align=cls_losses_align, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.OCEAN.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict


def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()


def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
