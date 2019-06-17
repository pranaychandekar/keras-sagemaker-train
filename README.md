
# ML Keras Training with Amazon SageMaker

## Tutorial

In this tutorial, we will work on building and training our custom Keras model in Amazon SageMaker. The tutorial is divided into 6 parts. From Part 1 to Part 5, we will implement the project on a CPU. Once we are confident, then in Part 6 we will see how we can now tweak this project to use a GPU. In this repository, I have defined a convention for project structure, filenames etc. to ensure reusability. You can clone this repository and perform minimal editing to get going with your own training logic. **Feel free to fork it or raise a pull request to make it more generic.** 

Lets dive into our tutorial.

---
### Part 1: Adding data to Amazon S3

Amazon Simple Storage Service or S3 is a storage service by AWS where we can store our data. Our SageMaker training job will download data from an S3 bucket, train the model and upload the trained model back into S3.

 1. Create an S3 bucket with the name '***keras-sagemaker-train***'
 2. Create two folders inside this bucket '***data***' and '***output***'
 3. Upload your data in the S3 bucket. For this tutorial, we will upload our data in Part 4.

***The S3 is now ready.***

---
### Part 2: Create a Notebook instance in SageMaker 
In this part, we will create one SageMaker Notebook instance which we will be using for building the docker image of our algorithm, testing the algorithm and initializing a training job. 

 1. Open the Amazon SageMaker console, click on '***Notebook instances***' and then click on '***Create notebook instance***'.  
<p align="center">
  <img src="/images/kst-01.png" alt="Amazon SageMaker console Notebook instance">
</p>

 2. For the instance type, select '***ml.t2.medium***'. Since we won’t be performing heavy operations, I chose a small instance. You can choose any instance as per your requirements. The details of different SageMaker instances are available [here](https://aws.amazon.com/sagemaker/pricing/instance-types/).
<p align="center">
  <img src="/images/kst-02.png" alt="Notebook instance type">
</p>

 3. Create a new IAM role by selecting '***Create a new role***' and select the options shown in the below image for the role configuration. Click ‘***Create role***’ to create a new role and then hit ‘***Create notebook instance***’ to submit the request for a new notebook instance.
<p align="center">
  <img src="/images/kst-03.png" alt="Amazon SageMaker new IAM Role">
</p>
 
It takes a few minutes for the Notebook instance to become available. The status will change from ‘***Pending***’ to ‘***InService’*** once the instance becomes available. In the meantime, let's change the policies of our IAM role. Our IAM role associated with the Notebook instance needs a read and write access to Amazon Elastic Conatiner Repository or ECR. We need this in order to push our algorithm docker images from our Notebook instance and for Amazon SageMaker training job instance can pull this image for training.
 1. Click on the name of the Notebook instance we just created. This will open a page with details about the Notebook instance.
<p align="center">
  <img src="/images/kst-04.png" alt="Notebook instance details page">
</p>

 2. From the details page, click on the IAM role associated with this instance. This will open a page with IAM role details.
<p align="center">
  <img src="/images/kst-05.png" alt="IAM Role details page">
</p>

 3.  Click on ‘***Attach policies***’ and then search for ‘***AmazonEC2ContainerRegistryFullAccess***’ policy, select it and then click on ‘***Attach policy***’. Please make sure to check the checkbox next to the policy before hitting `Attach policy`.
<p align="center">
  <img src="/images/kst-06.png" alt="Attach ECR policy IAM Role">
</p>

***The Notebook instance is now ready.***


---
### Part 3: Setting up the project
As a part of this tutorial, we will be using this very same repository. So let us clone this project on our Notebook instance. 

From the Amazon SageMaker console, click ‘***Open Jupyter***’ to navigate into the Jupyter notebook. Under ‘***New***’, select ‘***Terminal***’. This will open up a terminal session to your notebook instance.
<p align="center">
  <img src="/images/kst-07.png" alt="SageMaker Notebook Terminal">
</p>

Now run the following command in this terminal to clone the project.
```
cd ~/SageMaker/
git clone https://github.com/pranayc6/keras-sagemaker-train.git
```
Let's take a look at the project structure:
<p align="center">
  <img src="/images/kst-08.png" alt="Project Structure">
</p>

 Now let's compare the project structure inside '***test_dir***' to the directory structure inside training job instance:
<p align="center">
  <img src="/images/kst-09.png" alt="Training job directory structure">
</p>

Yes! I have replicated the directory structure of the training job instance inside the '***test_dir***'. This will make it easy to setup paths inside our training code and then test the docker image locally on the Notebook instance.

---
### Part 4: Local testing on Notebook instance
In this part, we will set up our '***test_dir***' by adding the data, upload this data to S3 for our training job reference, build docker image of our algorithm and test it on our Notebook instance.  

Let's begin with the data part:

 1. Switch to the data directory.
```
cd ~/SageMaker/keras-sagemaker-train/local_test/test_dir/input/data/training/
```
 2. Extract the data from the file '***data_set.tar.gz***'.
```
tar xvzf data_set.tar.gz
```
3. Upload the data to the S3 bucket.
```
aws s3 cp data_set s3://keras-sagemaker-train/data/data_set
```

***The data is now ready for local testing as well as for training job.***

Now let us build the docker image of our algorithm.

 1. Switch to the directory with the file '***Dockerfile.cpu***'.
```
cd ~/SageMaker/keras-sagemaker-train/
```
2. Change the access permissions of the train file.
```
chmod +x src/*
```
3. Build the docker image of your algorithm.
```
docker build -t keras-sagemaker-train:cpu -f Dockerfile.cpu .
```
4. Check whether the new docker image is available.
```
docker images
```
<p align="center">
  <img src="/images/kst-10.png" alt="List of docker images on Notebook instance">
</p>

***The docker image of our algorithm is now ready.***

It is now time to test our algorithm image locally.

 1. Switch to the directory with the local testing script.
```
cd ~/SageMaker/keras-sagemaker-train/local_test/
```
2. Run the training locally.
```
./train_local.sh keras-sagemaker-train:cpu
```
You can now see the algorithm in action.
<p align="center">
  <img src="/images/kst-11.png" alt="Local training 01">
</p>
<p align="center">
  <img src="/images/kst-12.png" alt="Local training 02">
</p>

We now have a saved model '***model.h5***' in the '***model***' directory.

***Congratulations! We had a successful local run.***

---
### Part 5: Running a training job
In this part, we will upload our algorithm image to Amazon ECR, define our training job and run it on a training instance. We will do all this in a jupyter-notebook.

Let us begin.
 1. Open ‘***keras-sagemaker-train.ipynb***’ from the home directory of this project.
<p align="center">
  <img src="/images/kst-13.png" alt="Jupyter-notebook">
</p>

2. Switch the notebook kernel from ‘***Kernel -> Change kernel***’ menu. Select '***conda_tensorflow_p36***'
<p align="center">
  <img src="/images/kst-14.png" alt="SageMaker notebook change kernel">
</p>

3. Please now follow the instructions in the jupyter-notebook for the remaining portion of this part and then get back here for the next part. See you soon.
<p align="center">
  <img src="/images/kst-15.png" alt="Waiting image">
</p>

If you have finished with the notebook then you can find the trained model sitting in the output folder of your S3 bucket.

---
### Part 6: Working with a GPU
The first 5 parts are actually build up for this part. We are now ready to run a training job on a GPU. In this part, we will make some minor changes to our existing project to do so. Since our algorithm is written using Keras we need not make any code changes to use a GPU. Keras will automatically detect the availability of the GPU and will use it. Though it is not necessary, I would recommend to use a GPU Notebook instance for this part. You can follow the steps in Part 2 to and select a GPU Notebook instance. I used ‘***ml.p2.xlarge***’ for this part. If you are not using a GPU instance you will not be able to perform a GPU test run on your Notebook instance.
 
***Perform the following exercise only if you are using a GPU Notebook instance else switch to the next exercise.***

1. Switch to the directory with the file '***Dockerfile.gpu***'.

 ```
cd ~/SageMaker/
```
2. Change the access permissions of the train file.
```
chmod +x src/*
```
3. Build the GPU docker image of your algorithm for local testing. 
```
docker build -t keras-sagemaker-train:gpu -f Dockerfile.gpu .
```
4. Run the training locally.
```
cd ~/SageMaker/keras-sagemaker-train/local_test/
./train_local_gpu.sh keras-sagemaker-train:gpu
```
***You can now see the algorithm in action on a GPU***:
<p align="center">
  <img src="/images/kst-16.png" alt="Local training on a GPU">
</p>

The wait is finally over. Let's run the training job on a GPU instance.

 1. Open the '***keras-sagemaker-train.ipynb***'.

 2. In Step 2, comment the line with docker build using ***Dockerfile.cpu*** and uncomment the one with the ***Dockerfile.gpu***.
<p align="center">
  <img src="/images/kst-17.png" alt="Jupyter-notebook GPU">
</p>

 3. In Step 7, change the train instance type to use a GPU. Though you can use any GPU instance, please refer to pricing before using it. They are costly. The complete list of the training instance pricing is available [here](https://aws.amazon.com/sagemaker/pricing/).
<p align="center">
  <img src="/images/kst-18.png" alt="Training job instance type">
</p>
 5. Run all the cells in the notebook.

***Congratulations for a successful GPU training job run!*** 
<p align="center">
  <img src="https://media.giphy.com/media/l0MYDGA3Du1hBR4xG/giphy.gif" alt="Training job instance type">
</p>


 ---
**Refernces:**

 1. [https://github.com/aws-samples/amazon-sagemaker-keras-text-classification](https://github.com/aws-samples/amazon-sagemaker-keras-text-classification)
 2. [https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb)

---

### License:

This repository is licensed under the Apache 2.0 License.
