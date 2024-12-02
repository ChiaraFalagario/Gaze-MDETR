import sys
import argparse
import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
import numpy as np
import requests
import torchvision.ops
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import csv
import time
from datetime import datetime
from skimage.measure import find_contours

from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from datasets import GazeMDETR_eval_util



# +++++++++++++++++++++++++++++++++++++++++++++++++++ Parser +++++++++++++++++++++++++++++++++++++++++++++++++++
"""
This section contains the definition of the different options to choose for running the code.
"""

parser = argparse.ArgumentParser(description='Caption category/details, evaluation mode/figure visualization and storage')
parser.add_argument('-k', '--k_value', type=int, default=3, help='Specify the k integer set to match the top-N recommendations objective for IoU evaluation')
parser.add_argument('-sf', '--save_figures', type=bool, default=True, help='Specify if you want to save the generated figures for heatmaps and final selections')
parser.add_argument('-vf', '--visualize_figures', type=bool, default=False, help='Specify if you want to visualize the generated figures for heatmaps and final selections')
parser.add_argument('-iou', '--iou_threshold', type=float, default=0.5, help='Specify the IoU threshold for the evaluation')
parser.add_argument('-sc', '--selection_criterion', type=str, choices=['iou','conf'], default='conf', help='Specify a criterion to select the bounding box.')
# USE TO TEST SINGLE CAPTIONS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#parser.add_argument('-cc', '--caption_category', type=str, choices=['A', 'B', 'C', 'D', 'E'], default='C', help='Specify a value (A, B, C, D, E) to determine the caption category. A:The, B:This is a, C:Look at the, D:Point at the, E:Pass the')
#parser.add_argument('-cd', '--caption_details', type=int, choices=[1, 2, 3, 4], default=1, help='Specify a detail level as (1, 2, 3, 4) to determine the caption details. 1:pose+color+name+placement, 2:pose+name+placement, 3:color+name, 4:name')
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
args=parser.parse_args()
# +++++++++++++++++++++++++++++++++++++++++++++++++++ Parser +++++++++++++++++++++++++++++++++++++++++++++++++++



# +++++++++++++++++++++++++++++++++++++++++++++++++++ Functions  +++++++++++++++++++++++++++++++++++++++++++++++++++
"""
This section contains the functions definitions for the code.
"""

torch.set_grad_enabled(False);


# Standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Normalize and resize norm_map, values between (0.2,1)
transform_normMap = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize((-0.25), (1.25))
])


# Process bounding boxes to adapt dimensions to image size
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Exclude bounding boxes that are too large
def filter_by_area(out_bbox, size):
    flag = []
    img_w, img_h = size
    img_area = img_w*img_h
    dim = out_bbox.size(0)
    t = out_bbox.numpy()

    for n in range(0,dim):
        x_l = t[n][0]
        y_l = t[n][1]
        x_r = t[n][2]
        y_r = t[n][3]
        w = x_r - x_l
        h = y_r - y_l
        box_area = w*h
        if box_area<(img_area/8):
            flag.append(n)

    return flag


# Define colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# Apply mask on image
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


# Plot the resulting bounding box on RGB image
def plot_results(pil_img, scores, boxes, labels, save_fig_path, masks=None):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]

    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)

    plt.imshow(np_image)
    plt.axis('off')
    if save_fig_path is not None:
        save_fig_dir = os.path.dirname(save_fig_path)
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)
        if args.save_figures:
            plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0.1)
        #if args.visualize_figures:
            # plt.show()
            # plt.close()
    else:
        #if args.visualize_figures:
            # plt.show()
            # plt.close()
        args.visualize_figures = False
            

# Print results
def add_res(results, ax, color='green'):
    if True:
        bboxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        
    colors = ['purple', 'yellow', 'red', 'green', 'orange', 'pink']

    for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
        cls_name = ll if isinstance(ll,str) else CLASSES[ll]
        text = f'{cls_name}: {ss:.2f}'
        print(text)
        ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))
# +++++++++++++++++++++++++++++++++++++++++++++++++++ Functions  +++++++++++++++++++++++++++++++++++++++++++++++++++       



# +++++++++++++++++++++++++++++++++++++++++++++++++++ Detection +++++++++++++++++++++++++++++++++++++++++++++++++++

model, postprocessor = torch.hub.load('ChiaraFalagario/Gaze-MDETR:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
model = model.cuda()
model.eval();


def plot_inference(im, caption, gaze, save_fig_path, gt_bbox):

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()

    # propagate through the model
    memory_cache = model(img, [caption], gaze, encode_and_save=True)
    outputs = model(img, [caption], gaze, encode_and_save=False, memory_cache=memory_cache)

    # keep only predictions with 0.1+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas >= 0.1).cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

    idx_filt = filter_by_area(bboxes_scaled, im.size) # im.size = width, height
    filt_boxes = bboxes_scaled[idx_filt]
    probas = probas[keep]
    probas = probas[idx_filt]

    # Extract the text spans predicted by each box
    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            span = memory_cache["tokenized"].token_to_chars(0, pos)
            predicted_spans [item] += " " + caption[span.start:span.end]

    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
    labels = [labels[i] if i < len(labels) else "" for i in idx_filt]
     
    
    iou_vals = torchvision.ops.box_iou(gt_bbox,filt_boxes)

    # +++++++++++++++++ Evaluate performance with IoU +++++++++++++++++
    if args.selection_criterion == 'iou':

        if iou_vals.nelement() != 0:
            iou_max_val = torch.max(iou_vals)
            index_iou_max_val = torch.argmax(iou_vals)

        else:
            iou_max_val = None
            index_iou_max_val = None

        if iou_max_val is not None:
            iou = iou_max_val.cpu().detach().numpy().astype(float)
            if iou >= args.iou_threshold:
                acc = 1
            else:
                acc = 0
            
            selected_bbox = filt_boxes[index_iou_max_val]
            selected_bbox = selected_bbox.unsqueeze(0)

            selected_label = [labels[index_iou_max_val]]

            selected_score = (probas)[index_iou_max_val]
            selected_score = selected_score.unsqueeze(0)

        else: 
            iou = 0.0
            acc = 0
            selected_bbox = torch.empty((0, 4))
            selected_score = torch.empty((0))
            selected_label = ""

    # +++++++++++++++++ Evaluate performance with Confidence +++++++++++++++++
    else:
        if len(probas)!=0:
            conf_max_val = torch.max(probas)
            idx_max_conf = torch.argmax(probas)
        else:
            conf_max_val = None
            idx_max_conf = None

        if conf_max_val is not None:
            iou = iou_vals[0][idx_max_conf].cpu().detach().numpy().astype(float)
            if iou >= args.iou_threshold:
                acc = 1
            else:
                acc = 0

            selected_bbox = filt_boxes[idx_max_conf]
            selected_bbox = selected_bbox.unsqueeze(0)

            selected_label = [labels[idx_max_conf]]

            selected_score = (probas)[idx_max_conf]
            selected_score = selected_score.unsqueeze(0)

        else: 
            iou = 0.0
            acc = 0
            selected_bbox = torch.empty((0, 4))
            selected_score = torch.empty((0))
            selected_label = ""
    
    plot_results(im, selected_score, selected_bbox, selected_label, save_fig_path)

    return acc, iou, selected_bbox, selected_label
# +++++++++++++++++++++++++++++++++++++++++++++++++++ Detection +++++++++++++++++++++++++++++++++++++++++++++++++++



# +++++++++++++++++++++++++++++++++++++++++++++++++++ Main +++++++++++++++++++++++++++++++++++++++++++++++++++
"""
This section contains the main code for the model demo. 

To run the demo, upload your files in the "annotated_data" folder.
"""

start = time.time_ns()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("start =", current_time)

# Folders names (adapt to your needs)
# ++++++++++++++++++++++++++++++++
file_path = "/home/falag/Desktop/new_analysis/annotated_test_data"
results_path = "/home/falag/Desktop/new_analysis/GazeMDETR_captions_tests"
metrics_path = "/home/falag/Desktop/new_analysis/GazeMDETR_captions_tests/Metrics_files"
# ++++++++++++++++++++++++++++++++

if os.path.exists(metrics_path)==False:
    os.makedirs(metrics_path)

sel_crit = args.selection_criterion

for cc in ('ABCDE'):
    # Open CSV to save metrics
    metrics_name = cc+'_session_metrics.csv'
    metrics_csv_path = os.path.join(metrics_path,metrics_name)

    if os.path.isfile(metrics_csv_path):
        metrics_csv = open(metrics_csv_path, 'a')
        csv_writer = csv.writer(metrics_csv)
    else: 
        metrics_csv = open(metrics_csv_path, 'w')
        csv_writer = csv.writer(metrics_csv)
        csv_header = ['Session',
                    'Image Path',
                    'Test code',
                    'Caption',
                    'Performance metric',
                    'Labels',
                    'Predicted bbox',
                    'Ground truth bbox',
                    'IoU value',
                    'Accuracy']
        csv_writer.writerow(csv_header)
    # Open CSV to save metrics
    
    for cd in range(1,5):
        print("Starting analysis for caption code: ", cc+'-'+str(cd)) 
        
        folders = sorted([f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))])
        
        for folder in folders:
            folder_path = os.path.join(file_path, folder)
            obj_folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
            
            for object in obj_folders:
                rgb_path = os.path.join(os.path.join(folder_path, object), 'rgb_image')
                normMap_path = os.path.join(os.path.join(folder_path, object), 'normMap')

                images = sorted([f for f in os.listdir(rgb_path) if '.xml' not in f])
                normMaps = sorted([f for f in os.listdir(normMap_path) if '.log' not in f])
                annotation_path = os.path.join(rgb_path, 'annotation.xml')

                tree = ET.parse(annotation_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    if obj.find('name').text != 'head':
                        obj_info = {
                            'name': obj.find('name').text,
                            'color': obj.find('color').text,
                            'pose': obj.find('pose').text,
                            'placement': obj.find('placement').text,
                            'bndbox': {
                                'xmin': float(obj.find('bndbox/xmin').text),
                                'ymin': float(obj.find('bndbox/ymin').text),
                                'xmax': float(obj.find('bndbox/xmax').text),
                                'ymax': float(obj.find('bndbox/ymax').text)
                            }
                        }
                gt_bbox = torch.tensor([[obj_info['bndbox']['xmin'], obj_info['bndbox']['ymin'], obj_info['bndbox']['xmax'], obj_info['bndbox']['ymax']]]) 

                for j in range(min(len(images),len(normMaps))): 
                    # Load the RGB images
                    im_path = os.path.join(rgb_path, images[j])
                    im = Image.open(im_path)
                    
                    # Define caption templates
                    caption_templates = {
                        'A': {
                            1: "The {pose} {color} {name} {placement}.",
                            2: "The {pose} {name} {placement}.",
                            3: "The {color} {name}.",
                            4: "The {name}.",
                        },
                        'B': {
                            1: "This is a {pose} {color} {name} {placement}.",
                            2: "This is a {pose} {name} {placement}.",
                            3: "This is a {color} {name}.",
                            4: "This is a {name}.",
                        },
                        'C': {
                            1: "Look at the {pose} {color} {name} {placement}.",
                            2: "Look at the {pose} {name} {placement}.",
                            3: "Look at the {color} {name}.",
                            4: "Look at the {name}.",
                        },
                        'D': {
                            1: "Point at the {pose} {color} {name} {placement}.",
                            2: "Point at the {pose} {name} {placement}.",
                            3: "Point at the {color} {name}.",
                            4: "Point at the {name}.",
                        },
                        'E': {
                            1: "Pass the {pose} {color} {name} {placement}.",
                            2: "Pass the {pose} {name} {placement}.",
                            3: "Pass the {color} {name}.",
                            4: "Pass the {name}.",
                        }
                    }

                    # Construct caption
                    caption_category = cc
                    caption_details = cd
                    caption = caption_templates[caption_category][caption_details].format(**obj_info)

                    caption_words = caption.split()
                   
                    save_fig_path = os.path.join(results_path,str(args.iou_threshold)+"_Bboxes_RGB",str(caption_category), str(caption_details), folder, object, "_".join(caption_words),images[j].split('.')[0])

                    # Load the heatmaps
                    norm_map_path = os.path.join(normMap_path, normMaps[j])
                    norm_map = Image.open(norm_map_path)

                    norm_map_gray = norm_map.convert('L')
                    normalized_norm_map_tensor = transform_normMap(norm_map_gray)

                    # Visualize norm_map tensor before downsampling
                    normalized_norm_map_tensor_array = np.squeeze(normalized_norm_map_tensor.cpu().numpy())*255
                    normalized_norm_map_tensor_image = Image.fromarray(normalized_norm_map_tensor_array.astype(np.uint8))
                    
                    # Visualize downsampled norm map
                    downsampled_norm_map = torch.nn.functional.interpolate(normalized_norm_map_tensor.unsqueeze(0),size=(25,34), mode='bilinear', align_corners=False).squeeze(0)
                    downsampled_norm_map_array = np.squeeze(downsampled_norm_map.cpu().numpy())*255
                    downsampled_norm_map_image = Image.fromarray(downsampled_norm_map_array.astype(np.uint8))
                    
                    save_normMap_path = os.path.join(results_path+"/Heatmaps",str(caption_category), str(caption_details), folder, object, "_".join(caption_words), images[j].split('.')[0])
                    
                    # Visualize heatmap tensor before and after downsampling
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].imshow(normalized_norm_map_tensor_image, cmap='gray')
                    axs[0].set_title("norm_map tensor")

                    axs[1].imshow(downsampled_norm_map_image, cmap='gray')
                    axs[1].set_title("norm_map tensor downsampled")
                    
                    if save_normMap_path is not None:
                        save_normMap_path_dir = os.path.dirname(save_normMap_path)
                        if not os.path.exists(save_normMap_path_dir):
                            os.makedirs(save_normMap_path_dir)
                        if args.save_figures:
                            plt.savefig(save_normMap_path, bbox_inches='tight', pad_inches=0.1)
                        #if args.visualize_figures:
                            #plt.show()
                            plt.close()
                    else:
                        #if args.visualize_figures:
                            #plt.show()
                            #plt.close()
                        args.visualize_figures = False

                    plt.close()
    
                    # Run the inference
                    accuracy, iou_val, pred_box, labels = plot_inference(im, caption, normalized_norm_map_tensor, save_fig_path, gt_bbox)
                    
                    # Write on csv file
                    csv_row = [folder,
                            im_path,
                            caption_category+"_"+str(caption_details),
                            caption,
                            sel_crit,
                            labels,
                            pred_box,
                            gt_bbox,
                            iou_val,
                            accuracy]
                    csv_writer.writerow(csv_row)

                
    metrics_csv.close()

end = time.time_ns()
secTaken = (end-start)*pow(10,-9)
m, s = divmod(secTaken, 60)
m_int = round(m)
s_int = round(s)
if len(str(s_int))==1:
    s_int = "0"+str(round(s))
print(f"Time taken:  {m_int}:{s_int}")
# +++++++++++++++++++++++++++++++++++++++++++++++++++ Main +++++++++++++++++++++++++++++++++++++++++++++++++++
