"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time
import nibabel as nib

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard
from utils.volume_stats import Dice3d, Jaccard3d, Sensitivity, Specificity #, F1_score
from networks.RecursiveUNet import UNet
# from networks.WideRecursiveUNet import wwUNet
from inference.UNetInferenceAgent import UNetInferenceAgent
from torch.nn import init
#from torch import nn

class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    The basic life cycle of a UNetExperiment is:

        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

        # Create data loaders
        # TASK: SlicesDataset class is not complete. Go to the file and complete it. 
        # Note that we are using a 2D version of UNet here, which means that it will expect
        # batches of 2D slices.
        
        train_df = dataset[split["train"]]
        val_df = dataset[split["val"]]
        test_df  = dataset[split["test"]]

        self.train_loader = DataLoader(SlicesDataset(train_df), batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(SlicesDataset(val_df), batch_size=config.batch_size, shuffle=True, num_workers=0)

        # we will access volumes directly for testing
        self.test_data = test_df

        # Do we have CUDA available?
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        else:
            print("GPU Status:",torch.cuda.is_available())
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure our model and other training implements
        # We will use a recursive UNet model from German Cancer Research Center, 
        # Division of Medical Image Computing. It is quite complicated and works 
        # very well on this task. Feel free to explore it or plug in your own model
        # num_classes=3, in_channels=1, initial_filter_size=64, 
        # kernel_size=3, num_downs=4, norm_layer=nn.InstanceNorm2d

        def kaiming_normal(p):
            class_names = p.__class__.__name__
            if class_names.find('Conv') != -1:
                init.kaiming_normal_(p.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif class_names.find('Linear') != -1:
                init.kaiming_normal_(p.weight.data)
        
        self.model = UNet(num_classes=3,initial_filter_size=256,num_downs=2) # 3 classes: {"0": "background", "1": "anterior", "2": "posterior"}
        #self.model = wwUNet(num_classes=3)
        #self.model.model.model[0][0]=nn.Conv2d(1,64,3,1,1)
        #self.model.model.model[0][1]=nn.InstanceNorm2d(num_features=64)
        #self.model.model.model[1][0]=nn.Conv2d(64,128,3,1,1)
        
        self.model.stemblock.conv1.apply(kaiming_normal)
        
        self.model.to(self.device)
        
        if self.device == "cuda":
            self.model = torch.nn.DataParallel(self.model)
            torch.backends.cudnn.benchmark = True

        # We are using a standard cross-entropy loss since the model output is essentially
        # a tensor with softmax'd prediction of each pixel's probability of belonging 
        # to a certain class
        self.loss_function = torch.nn.CrossEntropyLoss()

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # Scheduler helps us update learning rate automatically
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # Set up Tensorboard. By default it saves data into runs folder. You need to launch
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        # Loop over our minibatches
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # TASK: You have your data in batch variable. Put the slices as 4D Torch Tensors of 
            # shape [BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE] into variables data and target. 
            # Feed data to the model and feed target to the loss function
            # data = <YOUR CODE HERE>
            # data = data.to('cuda')
            data = batch['images'].to('cuda')
            
            # import matplotlib
            # import numpy as np
            # import matplotlib.pyplot as plt
            # %matplotlib inline  
            # data_np = data.cpu().detach().numpy()
            # plt.imshow(data_np[2][0]); plt.show()
            
            # target = <YOUR CODE HERE>
            target = batch['segs'].to('cuda')
            
            # prediction
            prediction = self.model(data)

            # We are also getting softmax'd version of prediction to output a probability map
            # so that we can see how the model converges to the solution
            prediction_softmax = F.softmax(prediction, dim=1)
            loss = self.loss_function(prediction, target[:, 0, :, :])

            # TASK: What does each dimension of variable prediction represent?
            # ANSWER: prediction = [number_of_images, classes, width_of_image, height_of_image]

            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                # Output to console on every 10th batch
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete")

                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                # You don't need to do anything with this function, but you are welcome to 
                # check it out if you want to see how images are logged to Tensorboard
                # or if you want to output additional debug data
                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter)

            print(".", end='')

        print("\nTraining complete")

    def validate(self):
        """
        This method runs validation cycle, using same metrics as 
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not 
        propagate
        """
        print(f"Validating epoch {self.epoch}...")

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            
            for i, batch in enumerate(self.val_loader):
                
                # TASK: Write validation code that will compute loss on a validation sample
                # <YOUR CODE HERE>
                data = batch['images'].to('cuda')
                target = batch['segs'].to('cuda')
                
                # required data shape: torch.Size([8, 1, 64, 64])
                # required target shape: torch.Size([8, 1, 64, 64])
                prediction = self.model(data)
                prediction_softmax = F.softmax(prediction, dim=1)
                loss = self.loss_function(prediction, target[:, 0, :, :])
                loss.requires_grad = True
                loss.backward()
                print(f"Batch {i}. Data shape {data.shape} Loss {loss}")

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax, 
            prediction,
            (self.epoch+1) * 100)
        
        print("\nValidation complete")

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")
        
        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        self.model.eval()
        # In this method we will be computing metrics that are relevant to the task of 3D volume
        # segmentation. Therefore, unlike train and validation methods, we will do inferences
        # on full 3D volumes, much like we will be doing it when we deploy the model in the 
        # clinical environment.

        # TASK: Inference Agent is not complete. Go and finish it. Feel free to test the class
        # in a module of your own by running it against one of the data samples

        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []
        sens_list = []
        spec_list = []
        # f1_list = []

        # for every in test set
        for i, x in enumerate(self.test_data):
            
            gt = x["seg"]   # test image ground truth        
            ti = x["image"] # test image data
            original_filename = x['filename'] # test image file name
            pred_filename = 'predicted_'+x['filename'] # test image file name
            
            file_path = os.path.join("../data","images",original_filename)
            
            original_images = nib.load(file_path)
            
            mask3d = np.zeros(ti.shape)
            pred = inference_agent.single_volume_inference(ti)
            mask3d = np.array(torch.argmax(pred, dim=1))

            # Save predicted labels to local environment for further verification 
            # with the original image NIFTI coordinate system
            pred_coord = nib.Nifti1Image(mask3d, original_images.affine)           
            pred_out_path = os.path.join("../data","preds")
            pred_out_file = os.path.join(pred_out_path,pred_filename)
            
            if not os.path.exists(pred_out_path):
                os.makedirs(pred_out_path)

            nib.save(pred_coord, pred_out_file)
            
            # We compute and report Dice and Jaccard similarity coefficients which 
            # assess how close our volumes are to each other

            # TASK: Dice3D and Jaccard3D functions are not implemented. 
            # Complete the implementation as we discussed
            # in one of the course lessons, you can look up definition of Jaccard index 
            # on Wikipedia. If you completed it
            # correctly (and if you picked your train/val/test split right ;)),
            # your average Jaccard on your test set should be around 0.80

            # a - prediction
            # b - ground truth
            # print("mask3d shape:", mask3d.shape, "; gt shape:", gt.shape)
            # print("print shapes")
            
            dc = Dice3d(mask3d, gt)
            dc_list.append(dc)
            
            jc = Jaccard3d(mask3d, gt)
            jc_list.append(jc)
            
            sens = Sensitivity(mask3d, gt)
            sens_list.append(sens)
            
            spec = Specificity(mask3d, gt)
            spec_list.append(spec)
            
            # f1 = F1_score(mask3d, gt)
            # f1_list.append(f1)
            
            # STAND-OUT SUGGESTION: By way of exercise, consider also outputting:
            # * Sensitivity and specificity (and explain semantic meaning in terms of 
            #   under/over segmenting)
            # * Dice-per-slice and render combined slices with lowest and highest DpS
            # * Dice per class (anterior/posterior)

            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc,
                "sensitivity": sens,
                "specificity": spec,
                # "f1": f1,
                })

            print(f"{x['filename']} Dice {dc:.4f}, Jaccard {jc:.4f}, Sensitivity {sens:.4f}, and Specificity {spec:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")
        
        avg_dc = np.mean(dc_list)
        avg_jc = np.mean(jc_list)
        avg_sens = np.mean(sens_list)
        avg_spec = np.mean(spec_list)
        # avg_f1 = np.mean(f1_list)
        
        out_dict["overall"] = {
            "mean_dice": avg_dc,
            "mean_jaccard": avg_jc,
            "mean_sensitivity": avg_sens,
            "mean_specificity": avg_spec,
            # "mean_f1": avg_f1,
            }

        print("\nTesting complete.")
        print("------------------------------")
        print(f"Average Dice {avg_dc:.4f}, Average Jaccard {avg_jc:.4f}, Average Sensitivity {avg_sens:.4f}, and Average Specificity {avg_spec:.4f}")
        
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        print("Experiment started.")

        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        # save model for inferencing
        self.save_model_parameters()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
