# Learning Control With Robust Control Contraction Metrics (RCCM) (Status: In active development)
Neural Network based implementation of the paper, "[Tube-Certified Trajectory Tracking for Nonlinear Systems
With Robust Control Contraction Metrics](https://arxiv.org/pdf/2109.04453.pdf)", by Pan Zhao

The learning framework and code base used in our work, is taken from,"[Learning Certified Control Using Contraction Metric](https://arxiv.org/abs/2011.12569)", by Dawei Sun, Susmit Jha, and Chuchu Fan.
For more information refer to the repository: [C3M](https://github.com/sundw2014/C3M)

## Dependencies
Dependencies include ```torch```, ```tqdm```, ```cvxpy```,```numpy```, and ```matplotlib```. You can install them using the following command.
```bash
pip install -r requirements.txt
```

## Usage
The ```main.py``` script can be used for learning the controller and RCCM certificate. Usage of this script is as follows
```
usage: main.py [-h] [--task TASK] [--no_cuda] [--bs BS]
               [--num_train NUM_TRAIN] [--num_test NUM_TEST]
               [--lr LEARNING_RATE] [--epochs EPOCHS] [--lr_step LR_STEP]
               [--lambda _LAMBDA] [--w_ub W_UB] [--w_lb W_LB] [--log LOG]

optional arguments:
  -h, --help                    show this help message and exit
  --task TASK                   Name of the model.
  --no_cuda                     Disable cuda.
  --bs BS                       Batch size.
  --num_train NUM_TRAIN
                                Number of samples for training.
  --num_test NUM_TEST           Number of samples for testing.
  --lr LEARNING_RATE            Base learning rate.
  --epochs EPOCHS               Number of training epochs.
  --lr_step LR_STEP
  --lambda _LAMBDA              Convergence rate: lambda
  --w_ub W_UB                   Upper bound of the eigenvalue of the dual metric.
  --w_lb W_LB                   Lower bound of the eigenvalue of the dual metric.
  --log LOG                     Path to a directory for storing the log.
  --ref_traj_many               Flag for generating many reference trajectories.
  --init_same                   Flag for initial conditions to be same for re and closed loop traj.
```

For example, run the following command to learn a controller for the 9-dimensional quadrotor model.
```
mkdir log_QUADROTOR_9D
python main.py --log log_QUADROTOR_9D --task QUADROTOR_9D --epochs 120 --lambda 1.5
```

Run the following command to evaluate the learned controller and plot the results.
```
python plot_o.py --pretrained_RCCM log_QUADROTOR_8D/controller_best.pth.tar --task QUADROTOR_9D --plot_type 3D --plot_dims 0 1 2
python plot_o.py --pretrained_RCCM log_QUADROTOR_8D/controller_best.pth.tar --task QUADROTOR_9D --plot_type error
```

