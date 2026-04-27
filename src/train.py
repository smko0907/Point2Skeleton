import os
import sys
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from GraphAE import LinkPredNet
from SkelPointNet import SkelPointNet
from DataUtil import PCDataset
from datetime import datetime
import FileRW as rw
import config as conf


def parse_args():
    parser = argparse.ArgumentParser(description='Point2Skeleton')
    parser.add_argument('--pc_list_file', type=str, default='../data/data-split/all-train.txt',
                        help='file of the names of the point clouds')
    parser.add_argument('--data_root', type=str, default='../data/pointclouds/',
                        help='root directory of all the data')
    parser.add_argument('--point_num', type=int, default=2000, help='input point number')
    parser.add_argument('--skelpoint_num', type=int, default=100, help='output skeletal point number')

    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--save_net_path', type=str, default='../training-weights/',
                        help='directory to save the network parameters')
    parser.add_argument('--save_net_iter', type=int, default=1000,
                        help='frequency to save the network parameters (number of iteration)')
    parser.add_argument('--save_log_path', type=str, default='../tensorboard/',
                        help='directory to save the training log (tensorboard)')
    parser.add_argument('--save_result_path', type=str, default='../log/',
                        help='directory to save the temporary results during training')
    parser.add_argument('--save_result_iter', type=int, default=1000,
                        help='frequency to save the intermediate results (number of iteration)')
    args = parser.parse_args()

    return args


def halve_learning_rate(optimizer, check_point, current_epoch, lr_init):
    if current_epoch == check_point:
        lr = lr_init * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def output_results(log_path, batch_id, epoch, input_xyz, skel_xyz, skel_r, A_init=None, A_final=None):
    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()

    if A_init is not None:
        A_init_save = A_init.detach().cpu().numpy()
    if A_final is not None:
        A_final_save = A_final.detach().cpu().numpy()

    for i in range(batch_size):

        save_name_input = log_path + str(batch_id[i]) + "_input.off"
        save_name_sphere = log_path + str(batch_id[i]) + "_sphere_" + str(epoch) + ".obj"
        save_name_center = log_path + str(batch_id[i]) + "_center_" + str(epoch) + ".off"
        
        save_name_A_init = log_path + str(batch_id[i]) + "_graph_init_" + str(epoch) + ".obj"
        save_name_A_final = log_path + str(batch_id[i]) + "_graph_final_" + str(epoch) + ".obj"
        
        rw.save_off_points(input_xyz_save[i], save_name_input)
        rw.save_spheres(skel_xyz_save[i], skel_r_save[i], save_name_sphere)
        rw.save_off_points(skel_xyz_save[i], save_name_center)

        if A_init is not None:
            rw.save_graph(skel_xyz_save[i], A_init_save[i], save_name_A_init)
        if A_final is not None:
            rw.save_graph(skel_xyz_save[i], A_final_save[i], save_name_A_final)


if __name__ == "__main__":
    #parse arguments
    args = parse_args()
    pc_list_file = args.pc_list_file
    data_root = args.data_root
    point_num = args.point_num
    skelpoint_num = args.skelpoint_num
    gpu = args.gpu
    save_net_path = args.save_net_path
    save_net_iter = args.save_net_iter
    save_log_path = args.save_log_path
    save_result_path = args.save_result_path
    save_result_iter = args.save_result_iter

    #create folders
    rw.check_and_create_dirs([save_net_path, save_log_path, save_result_path])

    #intialize networks
    model_skel = SkelPointNet(num_skel_points=skelpoint_num, input_channels=0, use_xyz=True)
    model_gae = LinkPredNet()
    optimizer_skel = torch.optim.Adam(model_skel.parameters(), lr=conf.LR_SPN)
    optimizer_gae = torch.optim.Adam(model_gae.parameters(), lr=conf.LR_GAE)

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    tb_writer = SummaryWriter(save_log_path + TIMESTAMP)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        print("GPU Number:", torch.cuda.device_count(), "GPUs!")
        model_skel.cuda()
        model_skel.train(mode=True)
        model_gae.cuda()
        # Set the module in training mode
        model_gae.train(mode=True)
    else:
        print("No CUDA detected.")
        sys.exit(0)

    #load data and train
    pc_list = rw.load_data_id(pc_list_file)
    train_data = PCDataset(pc_list, data_root, point_num)
    # -------------------------------------------------
    # Data Loader combines a dataset and a sampler; return an iterable over the given dataset
    # dataset: dataset from which to load the data; pointcloud data 
    # batch_size: how many samples per batch to load
    # num_workers: how many subprocesses to use for data loading 
    # shuffle: reshuffle the data at every epoch
    # drop_last: set to True to drop the last incomplete batcch if the dataset size is not divisble by batch size
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=conf.BATCH_SIZE, 
                              shuffle=True, 
                              drop_last=True)
    
    iter = -1
    total_epoch = conf.PRE_TRAIN_EPOCH + conf.SKELPOINT_TRAIN_EPOCH + conf.GAE_TRAIN_EPOCH
    
    for epoch in range(total_epoch):
        for k, batch_data in enumerate(train_loader):
            iter += 1
            print('epoch, iter:', epoch, iter)
            
            batch_id, batch_pc = batch_data
            batch_id = batch_id
            batch_pc = batch_pc.cuda().float()
            
            ######################################
            # pre-train skeletal point network
            ######################################
            if epoch < conf.PRE_TRAIN_EPOCH:
                print('######### pre-training #########')
                skel_xyz, skel_r, shape_features = model_skel(batch_pc, compute_graph=False)
                loss_pre = model_skel.compute_loss_pre(batch_pc, skel_xyz)

                optimizer_skel.zero_grad()
                loss_pre.backward()
                optimizer_skel.step()

                tb_writer.add_scalar('SkeletonPoint/loss_pre', loss_pre.item(), iter)
                if iter % save_result_iter == 0:
                    output_results(save_result_path, batch_id, epoch, batch_pc, skel_xyz, skel_r)
                if iter % save_net_iter == 0:
                    torch.save(model_skel.state_dict(), save_net_path + 'weights-skelpoint-pre.pth')

            ######################################
            # train skeletal point network with geometric losses
            ######################################
            elif epoch < conf.PRE_TRAIN_EPOCH + conf.SKELPOINT_TRAIN_EPOCH:
                print('######### skeletal point training #########')
                skel_xyz, skel_r, shape_features = model_skel(batch_pc, compute_graph=False)
                loss_skel = model_skel.compute_loss(batch_pc, skel_xyz, skel_r, None, 0.3, 0.4)

                optimizer_skel.zero_grad()
                loss_skel.backward()
                optimizer_skel.step()

                tb_writer.add_scalar('SkeletalPoint/loss_skel', loss_skel.item(), iter)
                if iter % save_result_iter == 0:
                    output_results(save_result_path, batch_id, epoch, batch_pc, skel_xyz, skel_r)
                if iter % save_net_iter == 0:
                    torch.save(model_skel.state_dict(), save_net_path + 'weights-skelpoint.pth')

            ######################################
            # train GAE
            ######################################
            else:
                print('######### GAE training #########')

                # 1. Frezee the skeletal point network
                # the network that finds the points (model_skel) is set to train(mode=False). 
                # This means we stop moving the points and focus only on connecting them.
                if epoch == conf.PRE_TRAIN_EPOCH + conf.SKELPOINT_TRAIN_EPOCH:
                    model_skel.train(mode=False)

                # 2. Get skeletal points and the node features from pointcloud batch:
                # if `compute_graph` is true, returns the initial graph called `A_init`, `valid_mask`, `known_mask`
                skel_xyz, skel_r, sample_xyz, weights, shape_features, A_init, valid_mask, known_mask = model_skel(
                    batch_pc, compute_graph=True)
                skel_node_features = torch.cat([shape_features, skel_xyz, skel_r], 2).detach()
                A_init = A_init.detach()

                # 3. Feeding the Network: We give the model_gae the point features and the initial graph.
                # Here, the forward action happens...
                # 1) Input: It takes the skeletal features (positions, radius) and the initial graph (A_init).
                # 2) Transformation: It passes them through the 12 layers of Graph Convolutions defined in the encode function.
                # 3) The Decoder: Finally, it runs the `InnerProductDecoder`, which calculates a "score" for every possible pair of points to see if they should be connected.
                A_pred = model_gae(skel_node_features, A_init)

                # 4. Calculating Loss: The "Loss" is the network's score on a test. 
                # It compares the connections the AI predicted to the "correct" connections.
                # If the AI misses a connection or adds a wrong one, the loss goes up.
                # This loss value tells the system exactly how much error exists in the current stage
                loss_MBCE = model_gae.compute_loss(A_pred, A_init, known_mask.detach())
                # MBCE (Modified Binary Cross Entropy)
                # 

                # 5. The Optimizer: 
                optimizer_gae.zero_grad()
                # loss_MBCE.backward() performs a mathmatical trick called backpropagation ("the blame game");
                # traveling backward through all 12 layers of the network to find out which
                # specific "neurons" (parameters) were responsible for the mistake
                loss_MBCE.backward()
                # The line optimizer_gae.step() is where the actual "learning" and "correction" happens. 
                # Given the "blame" information, the AI tweaks its internal settings (weights) 
                # to make the loss smaller the next time it tries.
                optimizer_gae.step()

                # 6. Recovering the adjacency Matrix:
                # Finally, recover_A takes the AI's "maybe" guesses (probabilities) and 
                # turns them into a hard "Yes" or "No" (1 or 0) for every possible connection.
                A_final = model_gae.recover_A(A_pred, valid_mask)

                if iter % save_result_iter == 0:
                    output_results(save_result_path, batch_id, epoch, batch_pc, skel_xyz, skel_r, A_init, A_final)
                if iter % save_net_iter == 0:
                    torch.save(model_gae.state_dict(), save_net_path + 'weights-gae.pth')
                tb_writer.add_scalar('GAE/loss_MBCE', loss_MBCE.item(), iter)
            
            
            '''
            ######################################
            # Train two networks jointly
            ######################################
            else:
                print('######### joint training #########')
                if epoch == conf.PRE_TRAIN_EPOCH + conf.SKELPOINT_TRAIN_EPOCH + conf.GAE_TRAIN_EPOCH:
                    model_skel.train(mode=True)

                # get skeletal points and node features
                skel_xyz, skel_r, sample_xyz, weights, shape_features, A_init, valid_mask, known_mask = model_skel(
                    batch_pc, compute_graph=True)
                skel_node_features = torch.cat([shape_features, skel_xyz, skel_r], 2).detach()

                # train GAE
                A_pred = model_gae(skel_node_features, A_init)
                loss_MBCE = model_gae.compute_loss(A_pred, A_init, known_mask.detach())

                halve_learning_rate(optimizer_gae, conf.LR_DROP_CHECKPOINT, epoch, conf.LR_GAE)
                optimizer_gae.zero_grad()
                loss_MBCE.backward()
                optimizer_gae.step()

                # get the adjacency matrix
                A_final = model_gae.recover_A(A_pred, valid_mask)
                loss_skel = model_skel.compute_loss(batch_pc, skel_xyz, skel_r, A_final, 0.3, 0.4, 0.1, lap_reg=True)

                halve_learning_rate(optimizer_skel, conf.LR_DROP_CHECKPOINT, epoch, conf.LR_SPN)
                optimizer_skel.zero_grad()
                loss_skel.backward()
                optimizer_skel.step()

                if iter % save_result_iter == 0:
                    output_results(save_result_path, batch_id, epoch, batch_pc, skel_xyz, skel_r, A_init, A_final)
                if iter % save_net_iter == 0:
                    torch.save(model_gae.state_dict(), save_net_path + 'weights-skelpoint-joint' + str(epoch) + '.pth')
                    torch.save(model_skel.state_dict(), save_net_path + 'weights-gae-joint' + str(epoch) + '.pth')

                tb_writer.add_scalar('GAE/loss_MBCE', loss_MBCE.item(), iter)
                tb_writer.add_scalar('SkeletalPoint/loss_skel', loss_skel.item(), iter)
            '''
            

