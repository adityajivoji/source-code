import matplotlib.pyplot as plt
import numpy as np
# InteractiveShell.ast_node_interactivity = "all"
# from tqdm import tqdm
# import plotly.express as px
import argparse
# from sklearn.model_selection import train_test_split
import os
import argparse
import torch
from supermarket.dataset.dataloader import Supermarket
from eth_ucy.dataloader_diverse import eth_dataset


import os
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import os

def plot_trajectory(_loc, _loc_end, _loc_pred_head, epoch, save_dir=None):
    prediction_head = 20
    plt.figure()
    for i in range(2):
        loc = _loc[i]
        loc_end = _loc_end[i]
        loc_pred_head = _loc_pred_head[i]
        
        for index in range(1):
            # Create a new figure for each plot
            loc_pred = loc_pred_head[index]
            # Extract x and y coordinates from loc, loc_end, and loc_pred
            x_loc, y_loc = zip(*loc)
            x_loc_end, y_loc_end = zip(*loc_end)
            x_loc_pred, y_loc_pred = zip(*loc_pred)

            # Generate more points to create smooth curves
            t_loc = np.arange(len(loc))
            t_loc_end = np.arange(len(loc_end))
            t_loc_pred = np.arange(len(loc_pred))

            t_loc_new = np.linspace(t_loc.min(), t_loc.max(), 300)
            t_loc_end_new = np.linspace(t_loc_end.min(), t_loc_end.max(), 300)
            t_loc_pred_new = np.linspace(t_loc_pred.min(), t_loc_pred.max(), 300)

            spl_loc = make_interp_spline(t_loc, np.column_stack((x_loc, y_loc)), k=3)
            spl_loc_end = make_interp_spline(t_loc_end, np.column_stack((x_loc_end, y_loc_end)), k=3)
            spl_loc_pred = make_interp_spline(t_loc_pred, np.column_stack((x_loc_pred, y_loc_pred)), k=3)

            loc_smooth = spl_loc(t_loc_new)
            loc_end_smooth = spl_loc_end(t_loc_end_new)
            loc_pred_smooth = spl_loc_pred(t_loc_pred_new)

            x_loc_smooth, y_loc_smooth = loc_smooth[:, 0], loc_smooth[:, 1]
            x_loc_end_smooth, y_loc_end_smooth = loc_end_smooth[:, 0], loc_end_smooth[:, 1]
            x_loc_pred_smooth, y_loc_pred_smooth = loc_pred_smooth[:, 0], loc_pred_smooth[:, 1]

            # Plot the trajectories
            plt.plot(x_loc_smooth, y_loc_smooth, color='blue', label='loc')
            plt.plot(x_loc_end_smooth, y_loc_end_smooth, color='green', label='loc_end')
            plt.plot(x_loc_pred_smooth, y_loc_pred_smooth, color='orange', label='loc_pred')

            # Scatter plot the original coordinates
            plt.scatter(x_loc, y_loc, color='blue')
            plt.scatter(x_loc_end, y_loc_end, color='green')
            plt.scatter(x_loc_pred, y_loc_pred, color='orange')

            # Connect the last point of loc to the first point of loc_end and loc_pred
            plt.plot([x_loc[-1], x_loc_end[0]], [y_loc[-1], y_loc_end[0]], color='blue')
            plt.plot([x_loc[-1], x_loc_pred[0]], [y_loc[-1], y_loc_pred[0]], color='blue')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Smooth Trajectory')
            plt.grid(True)
            plt.legend()

        # Save the plot if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'trajectory_plot_{epoch}_{index}.png')
        plt.savefig(save_path)
    plt.close()

def visualize(loader_test, model, epoch=None, save_dir=None):
    with torch.no_grad():
        for _, data in enumerate(loader_test):
            if data is not None:
                loc, loc_end, num_valid = data
                loc = loc.cuda().to(torch.float32)
                loc_end = loc_end.cuda().to(torch.float32)
                num_valid = num_valid.cuda()
                num_valid = num_valid.type(torch.int)

                vel = torch.zeros_like(loc)
                vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
                vel[:,:,0] = vel[:,:,1]

                vel = vel * 1
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
                loc_pred, category_list = model(nodes, loc.detach(), vel, num_valid)

                loc_pred = np.array(loc_pred.cpu()) # B,N,20,T,2 [:,0,:,:,:]
                loc_end = np.array(loc_end.cpu()) # B,N,T,2 [:,0,:,:]
                loc = np.array(loc.cpu().unsqueeze(1)) # B,N,T,2 [:,0,:,:]

                loc_end = loc_end[:,:,None,:,:]

                loc = loc.squeeze()
                loc_end = loc_end.squeeze()
                loc_pred = loc_pred.squeeze()
                index = np.random.randint(0,loc.shape[0])
                plot_trajectory(loc[index], loc_end[index],loc_pred[index], epoch, save_dir=save_dir)
                break
            
if  __name__ == "__main__":
    print("GPU COUNT", torch.cuda.device_count())
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--past_length', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--future_length', type=int, default=12, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before logging test')
    parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                        help='folder to output vae')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                        help='learning rate')
    parser.add_argument('--epoch_decay', type=int, default=2, metavar='N',
                        help='number of epochs for the lr decay')
    parser.add_argument('--lr_gamma', type=float, default=0.8, metavar='N',
                        help='the lr decay ratio')
    parser.add_argument('--nf', type=int, default=64, metavar='N',
                        help='learning rate')
    parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                        help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
    parser.add_argument('--attention', type=int, default=0, metavar='N',
                        help='attention in the ae model')
    parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                        help='number of layers for the autoencoder')
    parser.add_argument('--degree', type=int, default=2, metavar='N',
                        help='degree of the TFN and SE3')
    parser.add_argument('--channels', type=int, default=64, metavar='N',
                        help='number of channels')
    parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                        help='maximum amount of training samples')
    parser.add_argument('--dataset', type=str, default="nbody", metavar='N',
                        help='nbody_small, nbody')
    parser.add_argument('--sweep_training', type=int, default=0, metavar='N',
                        help='0 nor sweep, 1 sweep, 2 sweep small')
    parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                        help='timing experiment')
    parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                        help='timing experiment')
    parser.add_argument('--div', type=float, default=1, metavar='N',
                        help='timing experiment')
    parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                        help='normalize_diff')
    parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                        help='use tanh')
    parser.add_argument('--subset', type=str, default='eth',
                        help='Name of the subset.')
    parser.add_argument('--model_save_dir', type=str, default='eth_ucy/saved_models',
                        help='Name of the subset.')
    parser.add_argument('--scale', type=float, default=1, metavar='N',
                        help='dataset scale')
    parser.add_argument("--apply_decay",action='store_true')
    parser.add_argument("--res_pred",action='store_true')
    parser.add_argument("--supervise_all",action='store_true')
    parser.add_argument('--model_name', type=str, default='eth_ckpt_best', metavar='N',
                        help='dataset scale')
    parser.add_argument('--test_scale', type=float, default=1, metavar='N',
                        help='dataset scale')
    parser.add_argument("--test",action='store_true')
    parser.add_argument("--vis",action='store_true')
    args = parser.parse_args()
    args.num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == 'supermarket':
        from supermarket.model_t import EqMotion
        dataset_test = Supermarket(args.subset, args.past_length, args.future_length, device)
    elif args.dataset == 'eth':
        from eth_ucy.model_t import EqMotion
        dataset_test = eth_dataset(args.subset, args.past_length, args.future_length, args.test_scale, split='test', phase='testing')

    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                              num_workers=args.num_workers)

    model = EqMotion(in_node_nf=args.past_length, in_edge_nf=2, hidden_nf=args.nf, in_channel=args.past_length, hid_channel=args.channels, out_channel=args.future_length,device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)    
    if args.dataset == 'supermarket':
        model.load_state_dict(torch.load("./supermarket/saved_models/german_4_ckpt_best.pth"))
    elif args.dataset == 'eth':
        # Specify the path to the saved model checkpoint
        checkpoint_path = "./eth_ucy/saved_models/univ_ckpt_best.pth.tar"

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))

        # Extract the model state dictionary from the checkpoint
        state_dict = checkpoint['state_dict']

        # Load the state dictionary into the model
        model.load_state_dict(state_dict)
    
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader_test):
            if data is not None:
                loc, loc_end, num_valid = data
                loc = loc.cuda().to(torch.float32)
                loc_end = loc_end.cuda().to(torch.float32)
                num_valid = num_valid.cuda()
                num_valid = num_valid.type(torch.int)

                vel = torch.zeros_like(loc)
                vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
                vel[:,:,0] = vel[:,:,1]

                batch_size, agent_num, length, _ = loc.size()

                vel = vel * 1
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
                loc_pred, category_list = model(nodes, loc.detach(), vel, num_valid)

                loc_pred = np.array(loc_pred.cpu()) # B,N,20,T,2 [:,0,:,:,:]
                loc_end = np.array(loc_end.cpu()) # B,N,T,2 [:,0,:,:]
                loc = np.array(loc.cpu().unsqueeze(1)) # B,N,T,2 [:,0,:,:]

                loc_end = loc_end[:,:,None,:,:]

                loc = loc.squeeze()
                loc_end = loc_end.squeeze()
                loc_pred = loc_pred.squeeze()
                index = 5
                print(loc.shape, loc_end.shape, loc_pred.shape)

                plot_trajectory(loc[index], loc_end[index],loc_pred[index], epoch=0, save_dir='eth_ucy/saved_models/')
                break



