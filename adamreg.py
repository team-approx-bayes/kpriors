import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# For registering forward hooks
def update_input(self, input, output):
    self.input = input[0].data
    self.output = output


# Check if device of param is the same as old_param_device, and warn if not
def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # check if in same gpu
            warn = (param.get_device() != old_param_device)
        else:  # check if in cpu
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, this is currently not supported.')
    return old_param_device


# Convert from parameters to a matrix
def parameters_to_matrix(parameters):
    param_device = None
    mat = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        m = param.shape[0]
        mat.append(param.view(m, -1))
    return torch.cat(mat, dim=-1)


# Get parameter gradients as a vector
def parameters_grads_to_vector(parameters):
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        if param.grad is None:
            raise ValueError('Gradient not available')
        vec.append(param.grad.data.view(-1))
    return torch.cat(vec, dim=-1)


# The AdamReg optimiser. Started from torch.optim.adam
class AdamReg(Optimizer):

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, prior_prec=1e-3,
                 prior_prec_old=0., amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= prior_prec:
            raise ValueError("invalid prior precision: {}".format(prior_prec))
        if not 0.0 <= prior_prec_old:
            raise ValueError("invalid prior precision: {}".format(prior_prec_old))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, prior_prec=prior_prec,
                        prior_prec_old=prior_prec_old, amsgrad=amsgrad)
        super(AdamReg, self).__init__(model.parameters(), defaults)

        # State initialisation
        parameters = self.param_groups[0]['params']
        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()
        self.state['hessian_ggn'] = torch.zeros_like(self.state['mu'])
        self.state['step'] = 0
        self.state['exp_avg'] = torch.zeros_like(self.state['mu'])
        self.state['exp_avg_sq'] = torch.zeros_like(self.state['mu'])
        if amsgrad:
            self.state['max_exp_avg_sq'] = torch.zeros_like(self.state['mu'])

        # Additional for K-priors and Replay
        self.memory_labels = None
        self.model = model
        self.previous_weights = None
        self.prior_prec_old = None
        self.train_set_size = 0

    # Iteration step for this optimiser
    def step(self, closure_data, closure_memory=None, adaptation_method=None):

        parameters = self.param_groups[0]['params']
        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()
        mu = self.state['mu']
        self.total_datapoints_this_iter = 0

        # Normal loss term over current task's data
        if closure_data is not None:
            vector_to_parameters(mu, parameters)
            train_nll = closure_data()
            train_nll.backward()
            grad = parameters_grads_to_vector(parameters).detach()
        else:
            grad = torch.zeros_like(mu)
            train_nll = 0.

        # Multiply by train set size
        if self.train_set_size > 0:
            grad.mul_(self.train_set_size)
            self.total_datapoints_this_iter += self.train_set_size

        # Loss term over memory points (only if K-priors or Replay)
        if closure_memory is not None:
            # Forward pass through memory points
            preds = closure_memory()
            self.total_datapoints_this_iter += len(preds)

            # Softmax on output
            preds_soft = torch.softmax(preds, dim=-1)

            # Calculate the vector that will be premultiplied by the Jacobian, of size M x 2
            delta_logits = preds_soft.detach() - self.memory_labels

            # Autograd
            grad_message = torch.autograd.grad(preds, self.model.parameters(), grad_outputs=delta_logits)

            # Convert grad_message into a vector
            grad_vec = []
            for i in range(len(grad_message)):
                grad_vec.append(grad_message[i].data.view(-1))
            grad_vec = torch.cat(grad_vec, dim=-1)

            # Add to gradient
            grad.add_(grad_vec.detach())

            # Weight regularisation
            if adaptation_method == "K-priors" and self.prior_prec_old is not None:
                grad.add_(self.previous_weights, alpha=-self.prior_prec_old)

        # Add l2 regularisation
        if self.param_groups[0]['weight_decay'] != 0:
            grad.add_(mu, alpha=self.param_groups[0]['weight_decay'])

        # Divide by train set size
        if self.total_datapoints_this_iter > 0:
            grad.div_(self.total_datapoints_this_iter)

        # Update equations
        lr = self.param_groups[0]['lr']

        # Adam update
        exp_avg, exp_avg_sq = self.state['exp_avg'], self.state['exp_avg_sq']
        beta1, beta2 = self.param_groups[0]['betas']
        amsgrad = self.param_groups[0]['amsgrad']
        if amsgrad:
            max_exp_avg_sq = self.state['max_exp_avg_sq']

        self.state['step'] += 1
        bias_correction1 = 1 - beta1 ** self.state['step']
        bias_correction2 = 1 - beta2 ** self.state['step']

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.param_groups[0]['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.param_groups[0]['eps'])

        step_size = lr / bias_correction1
        mu.addcdiv_(exp_avg, denom, value=-step_size)
        vector_to_parameters(mu, parameters)

        return train_nll
