import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    return s - s*s


# Select memory points ordered by their h'(f) (==lambda) values (descending=True picks most important points)
def select_memory_points(dataloader, model, num_points, additional_memory_data=None, use_cuda=False, descending=True):

    memory_points_list = {}
    points_indices = {}

    # Data
    data, target = dataloader

    # Choose number of points per class correctly weighted
    num_points_per_class = [int(num_points/2),int(num_points/2)]
    if torch.sum(target==0) < num_points_per_class[0]:
        num_points_per_class[0] = torch.sum(target==0).numpy()
        num_points_per_class[1] = num_points - num_points_per_class[0]
    elif torch.sum(target==1) < num_points_per_class[1]:
        num_points_per_class[1] = torch.sum(target==1).numpy()
        num_points_per_class[0] = num_points - num_points_per_class[1]

    # Forward pass through all data
    if use_cuda:
        data_in = data.cuda()
    else:
        data_in = data

    preds = model.forward(data_in)

    # h'(f) (== lambda) on output
    lamb = softmax_hessian(preds)
    if use_cuda:
        lamb = lamb.cpu()
    lamb = torch.sum(lamb, dim=-1)
    lamb = lamb.detach()

    for cid in range(2):
        p_c = data[target == cid]
        indices_for_points = np.argwhere(target == cid)[0].numpy()
        if len(p_c) > 0:
            scores = lamb[target == cid]
            _, indices = scores.sort(descending=descending)
            memory_points_list[cid] = p_c[indices[:num_points_per_class[cid]]]
            points_indices[cid] = indices_for_points[indices[:num_points_per_class[cid]]]

    r_points = []
    r_labels = []
    r_indices = []
    for cid in range(2):
        r_points.append(memory_points_list[cid])
        r_labels.append(cid*torch.ones(memory_points_list[cid].shape[0], dtype=torch.long,
                                   device=memory_points_list[cid].device))
        r_indices.append(points_indices[cid])

    memory_points = {}
    memory_points['inputs'] = torch.cat(r_points, dim=0)
    memory_points['true_labels'] = torch.cat(r_labels, dim=0)
    if np.sum(num_points_per_class) > 2:
        memory_points['indices'] = np.concatenate(np.array(r_indices), axis=0)
    else:
        memory_points['indices'] = r_indices

    # If there is additional_memory_data, add that to memory_points['inputs']
    if additional_memory_data is not None:
        memory_points['inputs'] = torch.cat((memory_points['inputs'], additional_memory_data[0]))
        memory_points['true_labels'] = torch.cat((memory_points['true_labels'], additional_memory_data[1]))

    # Soft labels in K-priors
    if use_cuda:
        memory_points['inputs'] = memory_points['inputs'].cuda()
    memory_points['soft_labels'] = torch.softmax(model.forward(memory_points['inputs']), dim=-1)

    return memory_points


# Plot results with increasing memory size
def plot_increasing_past_size(test_accuracies, num_points_list, plot_title=None, path=None):

    # Plot
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(6, 6))
    axs = plt.subplot(1, 1, 1)

    for adaptation_method in test_accuracies:

        linestyle = 'solid'
        linewidth = 7
        marker = None
        if adaptation_method == "Replay":
            colour = 'b'
            marker = 'o'
            zorder = 2
        elif adaptation_method == "K-priors":
            colour = 'r'
            marker = 's'
            zorder = 3

        # Different random seeds
        accuracies_array = np.array(test_accuracies[adaptation_method]).reshape(-1, (len(num_points_list)))
        accuracies_mean = np.mean(accuracies_array, axis=0)
        accuracies_std = np.std(accuracies_array, axis=0)

        plt.plot(num_points_list, accuracies_mean, color=colour, linewidth=linewidth,
                 linestyle=linestyle, marker=marker, mfc='w', ms=17, mew=3, zorder=zorder, label=adaptation_method)

        if len(accuracies_array) > 1:
            plt.fill_between(num_points_list, accuracies_mean - accuracies_std, accuracies_mean + accuracies_std,
                             alpha=0.2, color=colour, zorder=zorder)



    # Batch result is exactly Replay with 100% of past data
    if num_points_list[0] == 1. and 'Replay' in test_accuracies:
        accuracies_array = np.array(test_accuracies['Replay']).reshape(-1, (len(num_points_list)))
        batch_test_accuracy = np.mean(accuracies_array, axis=0)[0]
        plt.plot(num_points_list, [batch_test_accuracy]*len(num_points_list), color='gray', linewidth=15,
                 linestyle='solid', zorder=1, label='Batch')

    plt.legend()
    plt.minorticks_off()
    plt.grid()
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    plt.xscale("log")
    plt.xticks(num_points_list, labels=[(int)(100*i) for i in num_points_list])

    plt.xlabel("Memory size (% of past data)")
    plt.ylabel("Validation acc (%)")
    if plot_title is not None:
        plt.title(plot_title)

    # Save figure if desired
    if path is not None:
        save_path = path + plot_title + '.pdf'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
