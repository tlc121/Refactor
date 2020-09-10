import tensorflow as tf
import numpy as np




def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious
    

def nms(bboxes, iou_threshold, k):
    #the shape of bboxes: [num_bboxes, 5]
    #the shape of bboxes: [num_bboxes, 1]
    best_bboxes = []
    temp_bboxes = bboxes.copy()
    while len(best_bboxes) <= k:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes.append(max_ind)
        iou_bboxes = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        iou_mask = iou > iou_threshold
        for idx in range(bboxes.shape[0]):
            if iou_mask[idx]:
                bboxes[idx, 4] = 0.0
                
                
    for idx in range(bboxes.shape[0]):
        if idx not in best_bboxes:
            bboxes[idx, 4] = 0.0
        else:
            bboxes[idx, 4] = temp_bboxes[idx, :]
     
    
    return bboxes

def iou(bbox1, bbox2):
    boxes1 = np.array(bbox1)
    boxes2 = np.array(bbox2)

    boxes1_area = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
    boxes2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    left_up       = np.maximum(boxes1[:2], boxes2[:2])
    right_down    = np.minimum(boxes1[2:], boxes2[2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[0] * inter_section[1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def topk_post_nms(bboxes, k):
    best_bboxes = []
    temp_bboxes = bboxes.copy()
    while len(best_bboxes) <= k:
        max_ind = np.argmax(bboxes[:, 4])
        best_bboxes.append(max_ind)
        bboxes[max_ind, 4] = 0.0
    
    for idx in range(bboxes.shape[0]):
        if idx not in best_bboxes:
            bboxes[idx, 4] = 0.0
        else:
            bboxes[idx, 4] = temp_bboxes[idx, :]
    return bboxes


def select_gtbbox(pred_bbox, label_bbox):
   #pred_bbox shape: [num_bboxes, 5+num_classes]
   #label_bbox shape: [num_bboxes, 5+num_classes]
    ret = np.zeros(shape=[pred_bbox.shape[0], 1], dtype=np.float32)
    for idx in range(pred_bbox.shape[0]):
        if label_bbox[idx, 4] > 0 and pred_bbox[idx, 4] > 0:
            pred_xywh = pred_bbox[idx, 0:4]
            label_xywh = label_bbox[idx, 0:4]
            iou_score = iou(pred_xywh, label_xywh)
            if iou_score > 0.3:
                ret[idx] = 1.0
    return ret
        
   
    
        
        
    
    
def postprocess(rpn_result_sbbox, rpn_result_mbbox, rpn_result_lbbox, pre_nms_top, post_nms_top, label_sbbox, label_mbbox, label_lbbox,threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    #rpn_result_bbox shape: [batch, h, w, 3, 5+num_classes]
    #first step: sort the matrix by the confidence score
    num_sbboxes = rpn_result_sbbox.shape[1]*rpn_result_sbbox.shape[2]*rpn_result_sbbox.shape[3]
    num_mbboxes = rpn_result_mbbox.shape[1]*rpn_result_mbbox.shape[2]*rpn_result_mbbox.shape[3]
    num_lbboxes = rpn_result_lbbox.shape[1]*rpn_result_lbbox.shape[2]*rpn_result_lbbox.shape[3]
    rpn_result_sbbox = np.reshape(rpn_result_sbbox, (rpn_result_sbbox.shape[0], num_sbboxes, rpn_result_sbbox[-1]))
    rpn_result_mbbox = np.reshape(rpn_result_mbbox, (rpn_result_mbbox.shape[0], num_mbboxes, rpn_result_mbbox[-1]))
    rpn_result_lbbox = np.reshape(rpn_result_lbbox, (rpn_result_lbbox.shape[0], num_lbboxes, rpn_result_lbbox[-1]))
    
#     conf_sbbox = rpn_result_sbbox[:, :, 4:5]
#     conf_mbbox = rpn_result_mbbox[:, :, 4:5]
#     conf_lbbox = rpn_result_lbbox[:, :, 4:5]
    
    for b in range(rpn_result_sbbox.shape[0]):
        rpn_result_sbbox[b] = nms(rpn_result_sbbox[b], iou_threshold, pre_nms_top)
        
    
    for b in range(rpn_result_mbbox.shape[0]):
        rpn_result_mbbox[b] = nms(rpn_result_mbbox[b], iou_threshold, pre_nms_top)
        
    for b in range(rpn_result_lbbox.shape[0]):
        rpn_result_lbbox[b] = nms(rpn_result_lbbox[b], iou_threshold, pre_nms_top)
    
    
    
    
    for b in range(rpn_result_sbbox.shape[0]):
        rpn_result_sbbox[b] = topk_post_nms(rpn_result_sbbox[b], post_nms_top)
        
    
    for b in range(rpn_result_mbbox.shape[0]):
        rpn_result_mbbox[b] = topk_post_nms(rpn_result_mbbox[b], post_nms_top)
        
        
    for b in range(rpn_result_lbbox.shape[0]):
        rpn_result_lbbox[b] = topk_post_nms(rpn_result_lbbox[b], post_nms_top)
       
    respond_rcnn_sbbox = np.zeros(shape=[rpn_result_sbbox.shape[0], num_sbboxes, 1])
    respond_rcnn_mbbox = np.zeros(shape=[rpn_result_mbbox.shape[0], num_mbboxes, 1])
    respond_rcnn_lbbox = np.zeros(shape=[rpn_result_lbbox.shape[0], num_lbboxes, 1])
    
    for b in range(rpn_result_sbbox.shape[0]):
        respond_rcnn_sbbox[b] = select_gtbbox(rpn_result_sbbox[b], label_sbbox[b])
    
    for b in range(rpn_result_mbbox.shape[0]):
        respond_rcnn_mbbox[b] = select_gtbbox(rpn_result_mbbox[b], label_mbbox[b])
        
    for b in range(rpn_result_lbbox.shape[0]):
        respond_rcnn_lbbox[b] = select_gtbbox(rpn_result_lbbox[b], label_lbbox[b])
        
    return rpn_result_sbbox, rpn_result_mbbox, rpn_result_lbbox, respond_rcnn_sbbox, respond_rcnn_mbbox, respond_rcnn_lbbox
        
        
    
    
    
 