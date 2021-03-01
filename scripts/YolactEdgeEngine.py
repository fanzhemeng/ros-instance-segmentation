from data import COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer
from utils.tensorrt import convert_to_tensorrt

from data import cfg, set_cfg
from utils.logging_helper import setup_logger

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import logging
import math


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(engine, argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default=None, type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', default=False, dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', default=False, dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', default=False, dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--eval_stride', default=5, type=int,
                        help='The default frame eval stride.')
    parser.add_argument('--output_coco_json', default=False, dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', default=False, dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', default=False, dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--fast_eval', default=False, dest='fast_eval', action='store_true',
                        help='Skip those warping frames when there is no GT annotations.')
    parser.add_argument('--deterministic', default=False, dest='deterministic', action='store_true',
                        help='Whether to enable deterministic flags of PyTorch for deterministic results.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--yolact_transfer', dest='yolact_transfer', action='store_true',
                        help='Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
    parser.add_argument('--coco_transfer', dest='coco_transfer', action='store_true',
                        help='[Deprecated] Split pretrained FPN weights to two phase FPN (for models trained by YOLACT).')
    parser.add_argument('--drop_weights', default=None, type=str,
                        help='Drop specified weights (split by comma) from existing model.')
    parser.add_argument('--calib_images', default=None, type=str,
                        help='Directory of images for TensorRT INT8 calibration, for explanation of this field, please refer to `calib_images` in `data/config.py`.')
    parser.add_argument('--trt_batch_size', default=1, type=int,
                        help='Maximum batch size to use during TRT conversion. This has to be greater than or equal to the batch size the model will take during inferece.')
    parser.add_argument('--disable_tensorrt', default=False, dest='disable_tensorrt', action='store_true',
                        help='Don\'t use TensorRT optimization when specified.')
    parser.add_argument('--use_fp16_tensorrt', default=False, dest='use_fp16_tensorrt', action='store_true',
                        help='This replaces all TensorRT INT8 optimization with FP16 optimization when specified.')
    parser.add_argument('--use_tensorrt_safe_mode', default=False, dest='use_tensorrt_safe_mode', action='store_true',
                        help='This enables the safe mode that is a workaround for various TensorRT engine issues.')
    #parser.set_defaults(no_hash=False)

    engine.args = parser.parse_args()

    if engine.args.output_web_json:
        engine.args.output_coco_json = True
    
    if engine.args.seed is not None:
        random.seed(args.seed)

class ImageResult:
    def __init__(self, classes, scores, boxes, mask, num_dets):
        self.classes = classes
        self.scores = scores
        self.boxes = boxes
        self.mask = mask
        self.num_dets = num_dets


class YolactEdgeEngine:

    def __init__(self):
        parse_args(self)
        self.args.config = 'yolact_edge_mobilenetv2_config'
        set_cfg(self.args.config)
        self.args.trained_model = '/home/ht/catkin_ws/src/instance_segmentation/scripts/weights/yolact_edge_mobilenetv2_124_10000.pth'
        self.args.top_k = 10
        self.args.score_threshold = 0.3
        self.args.trt_batch_size = 3
        self.args.disable_tensorrt = False
        self.args.use_fp16_tensorrt = False
        self.args.use_tensorrt_safe_mode = True
        self.args.cuda = True
        self.args.fast_nms = True
        self.args.display_masks = True
        self.args.display_bboxes= True
        self.args.display_text = True
        self.args.display_scores = True
        self.args.display_linecomb = False
        self.args.fast_eval = False
        self.args.deterministic = False
        self.args.no_crop = False
        self.args.crop = True
        self.args.calib_images = '/home/ht/catkin_ws/src/instance_segmentation/scripts/data/coco/calib_images'

        setup_logger(logging_level=logging.INFO)
        self.logger = logging.getLogger('yolact.eval')
        
        self.color_cache = defaultdict(lambda: {})

        with torch.no_grad():
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

            self.logger.info('Loading model...')
            self.net = Yolact(training=False)
            if self.args.trained_model is not None:
                self.net.load_weights(self.args.trained_model, args=self.args)
            else:
                self.logger.warning('No weights loaded!')
            self.net.eval()
            self.logger.info('Model loaded.')
            convert_to_tensorrt(self.net, cfg, self.args, transform=BaseTransform())

    def evaluate(self, train_mode=False, train_cfg=None):
        with torch.no_grad():
            self.net = self.net.cuda()
            self.net.detect.use_fast_nms = self.args.fast_nms
            cfg.mask_proto_debug = self.args.mask_proto_debug
            inp, out = self.args.images.split(':')
            self.evalimages(inp, out)

    def evalimages(self, input_folder:str, output_folder:str):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        print()
        for p in Path(input_folder).glob('*'):
            path = str(p)
            name = os.path.basename(path)
            name = '.'.join(name.split('.')[:-1]) + '.jpg'
            out_path = os.path.join(output_folder, name)

            img = cv2.imread(path)
            img_out = self.evalimage(img, out_path)
            #print(path + ' -> ' + out_path)
        print('Done.')

    def detect(self, img_in, return_imgs=False):
        with torch.no_grad():
            self.net = self.net.cuda()
            self.net.detect.use_fast_nms = self.args.fast_nms
            cfg.mask_proto_debug = self.args.mask_proto_debug
            #return self.evalimage(img_in[0])
            return self.evalbatch(img_in, return_imgs)

    def evalbatch(self, imgs, return_imgs=False):
        frame = torch.from_numpy(np.array(imgs)).cuda().float()
        batch = FastBaseTransform()(frame)

        if cfg.flow.warp_mode != 'none':
            assert False, 'Evaluating the image with a video-based model.'

        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}

        #start_time = time.time()
        preds = self.net(batch, extras=extras)["pred_outs"]
        #end_time = time.time()
        #print('%.3f s' % (end_time-start_time))

        imgs_out = []
        allres = []
        for i, img in enumerate(imgs):
            if return_imgs:
                img_out, res = self.prep_display(preds, frame[i], None, None, undo_transform=False, batch_idx=i, create_mask=True, return_imgs=return_imgs)
                imgs_out.append(img_out)
                allres.append(res)
            else:
                res = self.prep_display(preds, frame[i], None, None, undo_transform=False, batch_idx=i, create_mask=True, return_imgs=return_imgs)
                allres.append(res)
        if return_imgs:
            return imgs_out, allres
        else:
            return allres

    def evalimage(self, img, save_path=None):
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))

        if cfg.flow.warp_mode != 'none':
            assert False, 'Evaluating the image with a video-based model.'

        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}

        preds = self.net(batch, extras=extras)["pred_outs"]

        return self.prep_display(preds, frame, None, None, undo_transform=False, create_mask=True)
        #if save_path:
        #    cv2.imwrite(save_path, img_numpy)
        #return img_numpy, mask

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, batch_idx=0, create_mask=False, return_imgs=False):
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape
            #print(h, " ", w)

        with timer.env('Postprocess'):
            t = postprocess(dets_out, w, h, batch_idx, visualize_lincomb = self.args.display_linecomb,
                                            crop_masks = self.args.crop,
                                            score_threshold = self.args.score_threshold)
            torch.cuda.synchronize()

        with timer.env('Copy'):
            if cfg.eval_mask_branch:
                masks = t[3][:self.args.top_k]
            classes, scores, boxes = [x[:self.args.top_k].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.args.score_threshold:
                num_dets_to_consider = j
                break

        idx_fil = []
        for i in range(num_dets_to_consider):
            if cfg.dataset.class_names[classes[i]] == 'car' or cfg.dataset.class_names[classes[i]] == 'truck':
                idx_fil.append(i)
        num_dets_to_consider = len(idx_fil)

        if num_dets_to_consider == 0:
            # no detection found so just output original image
            if not create_mask:
                return (img_gpu * 255).byte().cpu().numpy()
            elif return_imgs:
                return (img_gpu * 255).byte().cpu().numpy(), ImageResult(None, None, None, np.zeros((h,w,1),dtype='uint8'), 0)
            else:
                return ImageResult(None, None, None, np.zeros((h,w,1),dtype='uint8'), 0)

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
            
            if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
                return self.color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    self.color_cache[on_gpu][color_idx] = color
                return color

        if self.args.display_masks and cfg.eval_mask_branch:
            # after this, mask is of size [num_dets, h, w, l]
            #masks = masks[:num_dets_to_consider, :, :, None]
            #classes = classes[:num_dets_to_consider]
            #scores = scores[:num_dets_to_consider]
            #boxes = boxes[:num_dets_to_consider, :]


            masks = masks[idx_fil, :, :, None]
            classes = classes[idx_fil]
            scores = scores[idx_fil]
            boxes = boxes[idx_fil, :]

            if create_mask:
                mask_img = np.zeros((h,w,1), dtype='uint8')
                for j in range(num_dets_to_consider):
                    mask_img += 10 * (j+1) * masks[j].cpu().numpy().astype(np.uint8)
                if not return_imgs:
                    return ImageResult(classes, scores, boxes, mask_img, num_dets_to_consider)

            # prepare the rgb image for each mask given their color (of size [num_dets, w, h, l])
            colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1,1,1,3) for j in range(num_dets_to_consider)], dim=0)
            masks_color = masks.repeat(1,1,1,3) * colors * mask_alpha

            # this is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1

            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        # then draw the stuff that needs to be done on cpu
        # note make sure this is a uint8 tensor or opencv will not anti aliaz text for wahtever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if self.args.display_text or self.args.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if self.args.display_bboxes:
                    cv2.rectangle(img_numpy, (x1,y1), (x2,y2), color, 1)

                if self.args.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = '%s: %.2f' % (_class, score) if self.args.display_scores else _class
                    text_pt = (x1, y1-3)
                    text_color = [255, 255, 255]

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return img_numpy, ImageResult(classes, scores, boxes, mask_img, num_dets_to_consider)


#if __name__ == '__main__':
#    engine = YolactEdgeEngine()
#    engine.evaluate()

