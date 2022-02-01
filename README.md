# Kpriors

Contains code for the NeurIPS 2021 paper by Khan and Swaroop, "[Knowledge-Adaptation Priors](https://arxiv.org/abs/2106.08769)".


Knowledge-adaptation priors (K-priors) reduce the cost of retraining by enabling quick and accurate adaptation for a wide-variety of tasks and models. This is made possible by a combination of weight and function-space priors to reconstruct the gradients of the past. 

The main file is ``main.py``, which can be run with various options. 
- The four different adaptation tasks can be changed by using the input variable ``adaptation_task``, which can be set to one of ``'add_data', 'remove_data', 'change_regulariser', 'change_model'``.
- For Figure 1 (right) in the paper, use ``dataset='usps_binary', network_type='MLP'`` (this is the default).
- For Figure 2(a) in the paper, use ``dataset='adult', network_type='Linear'``.
- For Figure 2(b) in the paper, use ``dataset='usps_binary', network_type='Linear'``.

Figures are saved in the main directory by default. Change this using the ``path`` variable. (Use ``path=None`` to view figures using matplotlib's ``plt.show()``). 

### Further details

The code was run with ``Python 3.7``, ``PyTorch v1.7.1``. For the full environment, see ``requirements.txt``. 

Hyperparameters are set at the top of the ``main.py`` file. The functional and weight regularisation is implemented as gradients in ``lbfgsreg.py`` and ``adamreg.py``; alternatively (and equivalently), one could use autograd over the loss function (see paper for the loss function). 

This code was written by Siddharth Swaroop (and the paper is joint work with Mohammad Emtiyaz Khan). Please raise issues here via github, or contact [Siddharth](ss2163@cam.ac.uk).

## Citation

```
@article{khan2021knowledge,
  title = {Knowledge-Adaptation Priors},
  author = {Khan, Mohammad Emtiyaz and Swaroop, Siddharth},
  journal = {Advances in Neural Information Processing Systems},
  year = {2021}
}
```
