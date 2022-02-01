import argparse
import torch
import numpy as np
import copy
from data_generators import UCIDataGenerator, USPSDataGenerator
import utils
import train
from models import LinearModel, MLP
from lbfgsreg import LBFGSReg
from adamreg import AdamReg


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='',
                    help='Path to where to store plots; use \'None\' if want default matplotlib view')
parser.add_argument('--seed_init', type=int, default=43, help='what random seed to use')
parser.add_argument('--num_runs', type=int, default=3, help='how many runs to do (each with a  different random seed)')
parser.add_argument('--dataset', type=str, default='usps_binary', help='what dataset to use: {adult, usps_binary}')
parser.add_argument('--network_type', type=str, default='MLP', help='what network type: {Linear, MLP}')
parser.add_argument('--adaptation_task', type=str, default='add_data',
                    help='which adaptation task: {add_data, remove_data, change_regulariser, change_model}')

args = parser.parse_args()

# Only consider use_cuda if MLP
use_cuda = False
if args.network_type == "MLP":
    use_cuda = True if torch.cuda.is_available() else False

# Which methods to run
adaptation_methods = ['Replay','K-priors']

# For storing and plotting test accuracies
test_accuracies_to_plot = {}
test_accuracies_to_plot['Replay'] = []
test_accuracies_to_plot['K-priors'] = []


# Settings for UCI Adult experiments
if args.dataset == "adult":
    polynomial_degree = 1
    remove_data_bool = False
    prior_prec = 5
    args.network_type = "Linear"  # Always Linear model with adult dataset
    learning_rate = 0.005
    num_epochs = 1000

    # What proportion of points to store; run many times
    fraction_points_stored_list = [1., 0.5, 0.2, 0.1, 0.07, 0.05, 0.02]

    # Settings for different adaptation tasks
    if args.adaptation_task == "remove_data":
        num_points_to_remove = 100  # These are chosen as the points with highest h'(f)
    elif args.adaptation_task == "change_regulariser":
        prior_prec_old = 50  # Reduce from 50 to 5 in adaptation task
        prior_prec = prior_prec_old
        prior_prec_new = 5


# Settings for UCI Adult experiments
if args.dataset == "usps_binary":
    remove_data_bool = False
    prior_prec = 50
    if args.network_type == "Linear":
        polynomial_degree = 1
        learning_rate = 0.1
        num_epochs = 300
    elif args.network_type == "MLP":
        polynomial_degree = None
        hidden_sizes = [100]  # MLP architecture
        learning_rate = 0.005
        num_epochs = 1000

    # What proportion of points to store; run many times
    fraction_points_stored_list = [1., 0.5, 0.2, 0.1, 0.07, 0.05, 0.02]

    # Settings for different adaptation tasks
    if args.adaptation_task == "change_regulariser":
        if args.network_type == "Linear":
            prior_prec_old = 50  # Reduce from 50 to 5 in adaptation task
            prior_prec = prior_prec_old
            prior_prec_new = 5
        elif args.network_type == "MLP":
            prior_prec_old = 5  # Reduce from 50 to 5 in adaptation task
            prior_prec = prior_prec_old
            prior_prec_new = 10

    if args.adaptation_task == "change_model" and args.network_type == "MLP":
        hidden_sizes = [100, 100]  # Go from two-hidden-layers to one-hidden-layer


# Repeat over many random seeds
for random_run in range(args.num_runs):

    seed = args.seed_init + random_run
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('')

    # Data generator
    if args.dataset == "adult":
        data_generator = UCIDataGenerator(adaptation_task=args.adaptation_task, seed=seed)
    elif args.dataset == "usps_binary":
        data_generator = USPSDataGenerator(adaptation_task=args.adaptation_task, polynomial_degree=polynomial_degree,
                                           seed=seed)

    # Load base task data
    base_train_data, base_test_data = data_generator.base_task_data()

    # Model and optimiser
    if args.network_type == "Linear":
        base_model = LinearModel(D_in=data_generator.dimensions, D_out=2)
        base_optimiser = LBFGSReg(base_model, lr=learning_rate, weight_decay=prior_prec)
    elif args.network_type == "MLP":
        base_model = MLP(D_in=data_generator.dimensions, hidden_sizes=hidden_sizes, D_out=2)
        base_model = base_model.cuda() if use_cuda else base_model
        base_optimiser = AdamReg(base_model, lr=learning_rate, weight_decay=prior_prec)
    else:
        raise ValueError("Incorrect network type: %s" % args.network_type)

    # Train on base task
    print('Training on base task...')
    train.train_model(base_model, base_optimiser, base_train_data, num_epochs=num_epochs, use_cuda=use_cuda)
    test_accuracy = train.test_model(base_model, base_test_data, use_cuda=use_cuda)
    print('Test accuracy on base task data: %f' % (test_accuracy))

    # Loop over fraction_points_stored_list for K-priors and Replay
    for num_points_counter in range(len(fraction_points_stored_list)):

        # Number of points to store for K-priors and Replay
        num_points_to_store = (int)(fraction_points_stored_list[num_points_counter]*data_generator.number_base_points)
        additional_memory_data = None

        # If remove_data task, then store the removed points too, for both K-priors and Replay
        if args.adaptation_task == "remove_data":
            if args.dataset == "adult":
                # Points to remove are picked by h'(f), so can simply add this number to num_points_to_store
                num_points_to_store += num_points_to_remove
            elif args.dataset == "usps_binary":
                # All of digit '8' is removed, so need to pick points that are not '8', and then add examples of '8' later
                base_train_data, _ = data_generator.data_split(digit_set=[0,1,2,3,4,5,6,7,9])
                additional_memory_data, _ = data_generator.data_split(digit_set=[8])

        # Select points
        memory_points = utils.select_memory_points(base_train_data, base_model, num_points_to_store,
                                                   additional_memory_data=additional_memory_data, use_cuda=use_cuda)

        # Load data for adaptation task
        adapt_train_data, adapt_test_data = data_generator.adaptation_task_data()

        # Train on adaptation task while regularising using K-priors or Replay
        for adaptation_method in adaptation_methods:

            # New model and optimiser
            if args.network_type == "Linear":
                model = copy.deepcopy(base_model)
                optimiser = LBFGSReg(model, lr=learning_rate, weight_decay=prior_prec)
                optimiser.previous_weights = base_model.return_parameters()
            elif args.network_type == "MLP":
                model = copy.deepcopy(base_model)
                model = model.cuda() if use_cuda else model
                optimiser = AdamReg(model, lr=learning_rate, weight_decay=prior_prec)
                optimiser.previous_weights = base_model.return_parameters()

            # Soft labels in K-priors, hard (true) labels in Replay
            if adaptation_method == "K-priors":
                memory_points['labels'] = memory_points['soft_labels']
            elif adaptation_method == "Replay":
                memory_points['labels'] = torch.nn.functional.one_hot(memory_points['true_labels'])

            # If change_model task, then need new model
            if args.adaptation_task == "change_model":
                if args.network_type == "Linear":
                    model = LinearModel(D_in=data_generator.dimensions, D_out=2)
                    optimiser = LBFGSReg(model, lr=learning_rate, weight_decay=prior_prec)

                    # Correct the memorable inputs to be of correct dimension as polynomial_degree has changed
                    if args.dataset == "adult":
                        adapt_train_inputs = torch.from_numpy(data_generator.X_train)
                        memory_points['inputs'] = adapt_train_inputs[memory_points['indices']]
                    elif args.dataset == "usps_binary":
                        adapt_train_data_interm,_ = data_generator.data_split(digit_set=[0,1,2,3,4,5,6,7,8,9])
                        memory_points['inputs'] = adapt_train_data_interm[0][memory_points['indices']]

                    optimiser.prior_prec_old = prior_prec

                    # Set correct previous_weights as polynomial_degree has changed
                    if args.dataset == "usps_binary":
                        num_parameters_poly1 = 257  # Poly degree 1 for USPS
                        num_parameters_poly2 = 33153  # Poly degree 2 for USPS
                    elif args.dataset == "adult":
                        num_parameters_poly1 = 124  # Poly degree 1 for Adult
                        num_parameters_poly2 = 7750  # Poly degree 2 for Adult
                    optimiser.previous_weights = torch.zeros(2 * num_parameters_poly1)
                    optimiser.previous_weights[:num_parameters_poly1] = base_model.upper.weight.data[0, :num_parameters_poly1]
                    optimiser.previous_weights[num_parameters_poly1:num_parameters_poly1 + num_parameters_poly1] = \
                        base_model.upper.weight.data[1, :num_parameters_poly1]

                    if use_cuda:
                        optimiser.previous_weights = optimiser.previous_weights.cuda()

                elif args.network_type == "MLP":
                    new_hidden_sizes = [100]
                    model = MLP(D_in=data_generator.dimensions, hidden_sizes=new_hidden_sizes, D_out=2)
                    model = model.cuda() if use_cuda else model
                    optimiser = AdamReg(model, lr=learning_rate, weight_decay=prior_prec)
                    optimiser.prior_prec_old = None

                else:
                    raise ValueError("Incorrect network type: %s" % args.network_type)

            # If change_regulariser task, then new prior_prec
            elif args.adaptation_task == "change_regulariser":
                optimiser.prior_prec_old = prior_prec_old
                prior_prec = prior_prec_new

            if args.adaptation_task == "remove_data":
                optimiser.prior_prec_old = prior_prec
                remove_data_bool = True

                # If Adult dataset, need to find points to remove (the points with highest h'(f))
                if args.dataset == "adult":
                    remove_points = utils.select_memory_points(base_train_data, base_model, num_points_to_remove, use_cuda=use_cuda)
                    adapt_train_data = (remove_points['inputs'], remove_points['true_labels'])

            if args.adaptation_task == "add_data":
                optimiser.prior_prec_old = prior_prec

            # Store past memory labels
            optimiser.memory_labels = memory_points['labels']

            # Train model
            print('Training on adaptation task using '+adaptation_method+' and fraction of past data of '+
                  str(fraction_points_stored_list[num_points_counter]))
            train.train_model(model, optimiser, adapt_train_data, num_epochs=num_epochs, memory_data=memory_points,
                              adaptation_method=adaptation_method, remove_data_bool=remove_data_bool, use_cuda=use_cuda)

            # Test model
            test_accuracy = train.test_model(model, adapt_test_data, use_cuda=use_cuda)
            test_accuracies_to_plot[adaptation_method].append(test_accuracy)
            print('Test accuracy on adaptation task data: %f' % (test_accuracy))


# Plot test accuracy figures
if len(adaptation_methods) > 1 or len(fraction_points_stored_list) > 1:
    plot_title = args.dataset+"_"+args.adaptation_task
    utils.plot_increasing_past_size(test_accuracies_to_plot, fraction_points_stored_list,
                                    plot_title=plot_title, path=args.path)
