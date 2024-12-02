# GazeMDETR: Multi-modal robotic architecture for object referring tasks aimed at designing new rehabilitation strategies

This is the repository to use GazeMDETR, combining gaze information in the original MDETR architecture.

![modarchitecture](https://github.com/user-attachments/assets/269e47fb-c532-4f23-9abd-fef3a3d66999)

## Abstract

The integration of robotics and Artificial Intelligence (AI) in healthcare applications holds significant potential for the development of innovative rehabilitation strategies. Great advantage of these new emerging technologies is the possibility to offer a rehabilitation plan that is personalised to each patient, especially in aiding individuals
with neurodevelopmental disorders, such as Autism Spectrum Disorder (ASD). In this context, a significant challenge is to endow robots with abilities to understand and replicate human social skills during interactions, while concurrently adapting to environmental stimuli. This extended abstract proposes a preliminary robotic architecture capable of estimating the human partner’s attention and recognizing the object to which the human is referring. Our work demonstrates how the robot’s ability to interpret human social cues, such as gaze, enhances system usability during object referring tasks.

## How to use GazeMDETR
Create a Conda virtual environment from the given ```gazemdetr_env.yml``` file in this repo using
```
conda env create -n my_env -f gazemdetr_env.yml
```
Change directory to your working directory and activate the environment with
```
conda activate my_env
```
Now you can run the ```GazeMDETR_demo.py``` script to obtain the predictions for a specified set of images. The test set you are using must be stored in the working directory too. 
You can customize the names for the test set and the results folders changing them in the script.

## Data

### GazeMDETRcluttered test set

The preliminary results were obtained using a specific set of images depicting human-robot interactions in a table-top scenario where the human partner gazes at different objects. The GazeMDETRcluttered set consists of 10267 frames, illustrating 4 participants in 3 different sessions and 15 different trials, gazing at different objects on the table. On the table were placed up to 11 objects chosen from the YCB dataset together with regular office objects, thus to increase the difficulty of the task. The participants were instructed to look at the requested object in a natural and spontaneous manner. For each session and for each trial, each object was gazed at for 5 seconds by the participant. The goal seeked in each recording sessionswas to recreate a cluttered scenario; in order to do this, each one of the three sessions was characterised by a specific arrangement of
objects (note that in a single session, the same object can be present multiple times):
1. Heterogeneous cluttered scenario
2. Scenario with only boxes
3. Scenario with only repeated objects

The tests were conducted on the GazeMDETRcluttered test set, which is stored in the iCubstor remote folder, under the name "GazeMDETR_data".
The data collected for the initial tests of MDETR and GazeMDETR and the outputs of the tests are accessible in the same folder, under the name "GazeMDETR_other" and then for each participant in the folder "0.RESULTS". 

## Original MDETR repository

**MDETR**: Modulated Detection for End-to-End Multi-Modal Understanding
Following you can find the links to the github page, the Colab project and the paper for the MDETR architecture.

[Website](https://ashkamath.github.io/mdetr_page/) • [Colab](https://colab.research.google.com/drive/11xz5IhwqAqHj9-XAIP17yVIuJsLqeYYJ?usp=sharing) • [Paper](https://arxiv.org/abs/2104.12763)

# Functioning of the architecture

## Combining gaze information with the MDETR data

The GazeMDETR architecture integrates visual attention information, through a **heatmap**, into the object detection process. Specifically, the model uses the heatmap to highlight relevant areas in the scene and integrates this information into the object detection pipeline, enhancing the model's effectiveness through visual attention. The main steps implemented are:

1. **Heatmap Preprocessing**:
   - The input images ($640\times480$ pixels) are compared with the **heatmaps** generated by the VTD module ($64\times64$). The heatmaps are then resized to match the image dimensions, using bilinear interpolation (```PyTorch interpolate```).
   - The heatmap is normalized in two steps: initially in the range $[0, 1]$, and then scaled to fall within the $[0.5, 1]$ range to give more weight to the "active" pixels in object detection.

2. **Integration into MDETR**:
   - The heatmap is multiplied by the input image, amplifying the most relevant pixels.
   - It is then flattened and fused with the **convolutional feature map** through direct multiplication, creating an enriched feature set with both visual and attention information.

3. **Optimization**:
   - The heatmap tensor is prepared for computation by moving it to the chosen processing unit (e.g., GPU) and sized for the batch.
  
## Examples of predictions
Following, are shown some of the predictions output from the model during the experiments on the GazeMDETRcluttered test set.

**Session 1, Participant 2**
![00000000](https://github.com/user-attachments/assets/f9efc1e6-b0d5-4b02-b1df-b45b943d318a) 
![00000000](https://github.com/user-attachments/assets/28de09cd-b021-4adb-bba5-119cddd0b8d9)

**Session 2, Participant 3**
![00000000](https://github.com/user-attachments/assets/952b2395-77af-4367-961f-3648cef57ee9)
![00000000](https://github.com/user-attachments/assets/3b3a5922-1d6a-44a7-869c-d2e34b827073)

**Session 3, Participant 1**
![00000000](https://github.com/user-attachments/assets/d369eb2d-b991-411e-bb49-266d2f90815e)
![00000000](https://github.com/user-attachments/assets/f4eb08d5-ee41-4601-8ba9-0a37084c5665)


## Annotating the collected dataset 
To ensure the collected data was suitable as input, the LabelImg tool was used to annotate the bounding boxes in all frames in the GazeMDETRcluttered test set.
Each annotation was saved in an XML file containing information such as the coordinates of the top-left and bottom-right corners of the bounding boxes, expressed in the format [$x_{min}$, $y_{min}$, $x_{max}$, $y_{max}$]. These XML files were then converted into TXT files containing information about the images and the annotated bounding boxes. The file includes the frame name, the save path, the source, the image dimensions, the annotated objects' information, and the positions of the bounding boxes for the head and observed object.

### Annotation Structure:
The XML file contains the following structure:

- Folder and Filename: Path to the annotated image.
- Source: Database source information.
- Size: Dimensions (width, height, depth) of the image.
- Objects: Each object (head and observed object) has:
	- Name: Object type (e.g., head, box).
	- Position and Placement: Additional metadata.
	- Bounding Box: The coordinates for the object’s bounding box.
- Bounding Box: Defined by xmin, ymin, xmax, ymax.

The structure is then converted into a TXT format with all relevant data for each frame.

## Prompts

The analysis was conducted using captions structured at various levels of detail to simulate a request during human interaction. These textual descriptions were designed to capture the fluidity, naturalness, and complexity of typical human expressions, enabling the model to understand and respond to requests formulated in a manner similar to how a human interlocutor would.

A total of 20 different captions were considered, summarized in the table below. Each caption is introduced with specific phrases or words and enriched with a varying number of attributes related to the referenced object. You can choose the type of prompt when running the code using the parser or run the code looping on all possible prompts.

**Phrases/Words:** Each caption starts with one of the following phrases
| Symbol  | Phrase/Word |
| :-------------: | :-------------: |
| A	| "The" |
| B	| "This is a" |
| C	| "Look at the" |
| D	| "Point at the" |
| E	| "Pass the" |

**Attributes:** Captions are enriched with different sets of attributes
| Symbol  | Attributes |
| :-------------: | :-------------: |
| 1	| Pose + Color + Name + Position |
| 2	| Pose + Name + Position |
| 3	| Color + Name |
| 4	| Name |

These captions simulate a range of interactions with different levels of detail to assess the model’s ability to understand human-like requests.


## Performance evaluation - Accuracy@1 Metric
The performance of **GazeMDETR** was evaluated and compared with MDETR using the *Accuracy@1* metric (denoted as Acc@1).
For each image, the predicted bounding box of the object was compared to the annotated ground truth by calculating the *Intersection over Union (IoU)*, also known as the *Jaccard index*, defined as:

$J(A, B) = \frac{|A \cup B|}{|A \cap B|}$

Where:
- \(A\) is the predicted bounding box.
- \(B\) is the ground truth bounding box.

### Thresholding and Accuracy Calculation

A threshold of **0.5** was set for the IoU. The prediction is considered as:

- **True Positive** (Acc@1 = 1) if the IoU between the predicted and ground truth bounding boxes is greater than or equal to 0.5.
- **False Positive** (Acc@1 = 0) if the IoU is less than 0.5.

Thus, the *Accuracy@1* metric measures the percentage of predictions where the predicted bounding box correctly overlaps with the ground truth (IoU >= 0.5).
The experiments revealed the following results:
| Session | GazeMDETR [Acc@1] | MDETR [Acc@1] |
| :-------------: | :-------------: | :-------------: |
| 1 | **0.82** | 0.64 |
| 2 | **0.51** | 0.34 |
| 3 | **0.41** | 0.24 |

## Application of the architecture in a rehabilitation scenario

In this section, the changes made to the GazeMDETR pipeline for evaluating its performance in real-world rehabilitation settings are discussed.

### Qualitative Evaluation
Instead of a quantitative analysis, a qualitative evaluation was performed to understand the model's behavior in real-life scenarios and optimize its performance. This approach helps identify common errors, assess the system's robustness under varying conditions (e.g., object angles, colors, and sizes), and test the model's ability to generalize to new scenarios.
The experiments revealed the following results, mediating the values of Acc@1 for 16 subjects:
| GazeMDETR [Acc@1] | MDETR [Acc@1] |
| :-------------: | :-------------: |
| **0.66** | 0.34 |

### Human Attention Estimation Module
The original pipeline relied on Face Detection using OpenPose to estimate head bounding boxes. However, in rehabilitation settings, where children are often accompanied by other people, OpenPose struggled to isolate the child’s face. As a result, the face detection and pose estimation modules were replaced with manually annotated head bounding boxes, which were then used for offline processing in the fine-tuned VTD module.

### Object Detection Module
The object detection module was modified to accommodate the different data structure in the new dataset. Changes included:
- Max Confidence criterion for bounding box selection, focusing on reliable results rather than strict quantitative evaluation.
- A modification to ensure the models detected smaller objects (cubes) in broader frames, as objects in rehabilitation scenes were smaller compared to those in the test set.
- IoU calculations and other quantitative metrics were removed in favor of a more qualitative visual evaluation to better interpret the model's performance in real-world scenarios.

These changes reflect the shift towards more interpretive, practical analysis for real-world applications, especially in rehabilitation settings.
