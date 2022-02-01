import torch
import torch.nn as nn


# Train model using optimiser on data for num_epochs
def train_model(model, optimiser, training_data, num_epochs, memory_data=None,
                adaptation_method=None, remove_data_bool=False, use_cuda=False):

    # Criterion for loss
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()

    # Train for num_epochs
    model.train()
    for epoch in range(num_epochs):

        # Closure over training data
        if training_data is not None:
            inputs, labels = training_data
            optimiser.train_set_size = len(inputs)

            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            def closure_main():
                optimiser.zero_grad()
                logits = model.forward(inputs)
                loss = criterion(torch.squeeze(logits, dim=-1), labels)

                # For removing data (instead of adding data)
                if remove_data_bool:
                    loss = -loss
                    # optimiser.total_datapoints_this_iter -= 2*optimiser.train_set_size

                return loss
        else:
            closure_main = None

        # Closure over datapoints in memory (for K-priors and Replay only)
        if memory_data is not None:
            def closure_memory():
                memory_inputs = memory_data['inputs']
                if use_cuda:
                    memory_inputs = memory_inputs.cuda()
                    optimiser.memory_labels = optimiser.memory_labels.cuda()
                    if optimiser.previous_weights is not None:
                        optimiser.previous_weights = optimiser.previous_weights.cuda()

                optimiser.zero_grad()
                logits = model.forward(memory_inputs)

                return logits
        else:
            closure_memory = None

        # Take an optimiser step
        train_nll = optimiser.step(closure_data=closure_main, closure_memory=closure_memory,
                                   adaptation_method=adaptation_method)

        # Print during training if desired
        print_during_training = False
        if print_during_training and epoch % 100 == 0:
            print('Epoch[%d]: Train nll: %f' % (epoch + 1, train_nll))


# Test model on testing_data, return test accuracy
def test_model(model, testing_data, use_cuda=False):

    correct = 0
    with torch.no_grad():
        model.eval()

        # Test data inputs and labels
        inputs, labels = testing_data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Find predictions from model
        logits = model.forward(inputs)

        # Calculate predicted classes
        pred = logits.data.max(1, keepdim=True)[1]

        # Count number of correctly predicted datapoints and calculate test accuracy
        correct += pred.eq(labels.data.view_as(pred)).sum()
        test_accuracy = 100.0 * float(correct) / len(inputs)

    return test_accuracy
