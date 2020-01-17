import os
import glob
import numpy as np
class logger(object):
    def __init__(self, path):
        self.path = os.path.join(os.getcwd(), "log/" + path + ".log")
        self.sample_path = os.path.join(os.getcwd(), "log/" + path + "_samples.log")
        
    def refresh(self):
        files = glob.glob(self.path.split(".")[0]+"*")
        for file in files:
            os.remove(file)
        
    def log(self, train_loss, train_acc,  val_loss, val_acc, epoch, train_pred, val_pred):
        result_init = "\n\nEpoch: {} \n".format(epoch)
        result_string_train = "train loss: {} | train acc: {}\n".format(train_loss, train_acc)
        result_string_val = "Validation loss: {} | Validation acc: {}\n".format(val_loss, val_acc)
        result_string = result_init+ result_string_train + result_string_val
        with open(self.path, "a+") as file:
            file.write(result_string)
        samples = 4
        train_idx = np.random.choice(np.arange(len(train_pred)), samples)
        train_samples = ""
        for i in train_idx:
            train_samples += "question: {} \n {} \n {}\n".format(train_pred[i][0], train_pred[i][1], train_pred[i][2])
        
        val_idx = np.random.choice(np.arange(len(val_pred)), samples)
        val_samples = ""
        for i in val_idx:
            val_samples += "question: {} \n {} \n {}\n".format(val_pred[i][0],val_pred[i][1],val_pred[i][2])
        result_string = result_init + "Train\n"+ train_samples + "Val\n" +val_samples + "\n"
        with open(self.sample_path, "a+") as file:
                file.write(result_string)