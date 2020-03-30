import sys, os
import argparse

parser = argparse.ArgumentParser(
    description='Train a new network on a data set')

# Compulsory command: python train.py data_directory
parser.add_argument('data_directory', action="store",type = str,help="Set a data directory to train the model on")

# Option 1 : Set directwersory to save checkpoints: python train.py data_dir --save_dir save_directory
parser.add_argument('--save_dir', action="store", type = str, dest = "save_dir", help="Set a directory to save the checkpoint")

# Option 2: Choose architecture: python train.py data_dir --arch "vgg13"
parser.add_argument('--arch', action="store", default = "models.vgg13", type = str, dest="model", help="Choose an architecture for the model like models.vgg16")

# Option 3: Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
parser.add_argument('--learning_rate', action="store", default = 0.001, type = int, dest = "learning_rate", help="Choose a learning rate")
parser.add_argument('--hidden_units', action="store", default = 2000, type = int, dest="num_hidden_layers",help="Choose the number of hidden units")
parser.add_argument('--epochs', action="store", default = 1, type = int, dest="epochs_number", help="Choose the number of epochs")

# Option 4: Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action="store", default = "cuda", type = str, dest="device", help="Choose the device cuda or cpu")

results = parser.parse_args()
#print(parser.parse_args(['-a', '-bval', '-c', '3']))

print('checkpoint_dir= {!r}'.format(results.save_dir))
print('architecture_model= {!r}'.format(results.model))
print('learning_rate        = {!r}'.format(results.learning_rate))
print('hidden_units      = {!r}'.format(results.num_hidden_layers))
print('epochs = {!r}'.format(results.epochs_number))
print('gpu = {!r}'.format(results.device))

os.system("python3 model.py "+str(results.device)+" "+str(results.model)+" "+str(results.learning_rate)+" "+str(results.num_hidden_layers)+" "+str(results.epochs_number)+" "+str(sys.argv[1]))
# Print : Prints out training loss, validation loss, and validation accuracy as the network trains

