# Module 1 - Running the exercises

This module is divided in four exercises: Warm up, Data Visualization, Model Building and Fleet Simulation. To run the exercises:  
  1. Open SageMaker Studio;
  2. Double-click the project you've created with **SageMaker Edge Manager WindTurbine Workshop** template;
  3. Click on the tab **Repositories**;
  4. Click on the **Local path** link of your repo (clone it if needed);
  5. Double-click on the directory **notebooks** and you will see the following exercices:

## 0/4 - Warm up
<a href="00-Warmup/00-Warmup.ipynb">Notebook</a>: In this notebook you will learn how to prepare a minimalist environment to run SageMaker Edge Agent and invoke its API using Python3; (Select SageMaker Studio Kernel: Data Science)

<p align="center">
    <img src="../imgs/EdgeManagerWorkshop_MinimalistArchitecture.png"></a>
</p>

## 1/4 - Data Visualisation
<a href="01-Data-Visualization/01-Data-Visualization.ipynb">Notebook</a>: In this notebook you will take a look on the raw data from the sensors, the selected features for training the model and how to prepare the dataset;

## 2/4 - Model Building

<a href="02-Training/02-Training-with-Pytorch.ipynb">Notebook</a>: How to create a ML Pipeline using SageMaker Model Building to: prepare the data, train the model & run batch prediction. The following diagram, shows the ML Pipeline and the other tasks you'll execute in this exercise.

<p align="center">
    <img src="../imgs/ggv2_lab2_train_pipeline.png"></a>
</p>

## 3.1/4 - Packaging model and Deploying the application using Greengrass V2
<a href="03-Package-Deploy/greengrass-v2/03-package-using-ggv2.ipynb">Notebook</a>: Here you'll complie the model using SageMaker Neo, package the model as a greengrass component using SageMaker Edge Manager and deploy the model and the application on a fleet of edge devices. The following diagram shows an overview of th architecture:
<p align="center">
    <img src="../imgs/ggv2_lab3_main_arch.png"></a>
</p>

## 3.2/4 - Packaging model and Deploying the application using IoT Jobs
<a href="03-Package-Deploy/iot-jobs/03-package-using-iot-jobs.ipynb">Notebook</a>: Here you'll complie the model using SageMaker Neo, package the model using SageMaker Edge Manager and deploy the model and the application using native AWS IoT deployment jobs. Below is the architecture diagram showcasing the same: 
<p align="center">
    <img src="../imgs/EdgeManagerWorkshop_Macro.png"></a>
</p>


## 4.1/4 - Run the fleet with IoT greengrass devices
<a href="04-Run-Fleet/greengrass-v2/04-run-fleet-ggv2.ipynb">Notebook</a>: Here you'll run the simulator of the wind turbine farm which sends the data to greengrass core devices via IoT MQTT. The core devices running the anomaly detection application detects if its anomaly or not and send the result back to the simulator via IoT MQTT.You'll be able to visualize all this process in the Simulator's UI.The following diagram shows the application architecture. The application is divided into two parts: Device Application and Simulator.
<p align="center">
    <img src="../imgs/ggv2_lab4_app_arch.png"></a>
</p>

If using Greengrass V2, we dont need an OTA module to update model as that will be handled by Greengrass itself.


## 4.2/4 - Packaging model and Deploying the application using IoT Jobs
<a href="04-Run-Fleet/iot-jobs/04-run-fleet-iot-jobs.ipynb">Notebook</a>: Here you'll run the simulator of the wind turbine farm and launch 5 SageMaker Edge Agents (one per turbine). A Python application then reads the sensors data, prepares the data and then invokes the anomaly detection model by calling SageMaker Edge Manager's API. You'll be able to visualize all this process in the Simulator's UI. The following diagram shows the application architecture. The application is divided into two parts: Device Application and Simulator.
<p align="center">
    <img src="../imgs/EdgeManagerWorkshop_App.png"></a>
</p>

This is an **standalone** setup for SageMaker Edge Manager. It means that no other service like will be used to build the edge device solution. In this case, the application that runs on the edge device needs to do some key tasks like Deploying/Updating the model. 

In the image bellow you can see the Python application (OTAModelUpdater) that communicates with an MQTT Topic is reponsible for receiving the **new model message** from the IoT Job and deploy the model to the edge device. The model package is downloaded from an S3 bucket and verified by the SageMaker Edge Agent.

<p align="center">
    <img src="../imgs/EdgeManagerWorkshop_Deployment.png"></a>
</p>

### Device application
This is a Python application that could be deployed into a real device. It is responsible for reading the sensors (from the simulator in this case), prepare the data and invoke the model through SageMaker Edge Agent client (protobufer stubs, generated from the **.proto** file provided by the agent) and updates the Simulator Dashboard. It has the infinite loop that process the anomalies for all the Wind Turbines.

### Simulator
This is a multithreaded application that launches one Thread per each Turbine and Each edge device to simulate a real environment, where you would have one Edge device connected to each Wind Turbine. It also has a dashboard with animation and some buttons, that you can see [here](../README.md).

The buttons serve to inject noise into the read data and simulates anomalies that are detected by the ML Model and plotted as violations in the Dashboard.

### Cleaning
In notebook #3, there are instructions of how to delete the project and all the resources created for it. Just run the cell and wait for the cleanup.
