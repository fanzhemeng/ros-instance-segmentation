from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer

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
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--eval_stride', default=5, type=int,
                        help='The default frame eval stride.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
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

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

    engine.args = parser.parse_args()

    if engine.args.output_web_json:
        engine.args.output_coco_json = True
    
    if engine.args.seed is not None:
        random.seed(args.seed)


class YolactEdgeEngine:

    def __init__(self):
        parse_args(self)
        self.args.config = 'yolact_edge_resnet50_config'
        set_cfg(self.args.config)
        self.args.trained_model = '/home/ht/catkin_ws/src/instance_segmentation/scripts/weights/yolact_edge_resnet50_54_800000.pth'
        self.args.top_k = 100
        self.args.cuda = True
        self.args.fast_nms = True
        self.args.display_masks = True
        self.args.display_bboxes= True
        self.args.display_text = True
        self.args.display_scores = True
        self.args.display_linecomb = False
        self.args.benchmark = False
        self.args.fast_eval = False
        self.args.deterministic = False
        self.args.no_sort = False
        self.args.mask_proto_debug = False
        self.args.no_crop = False
        self.args.images = '../detect:../detect-output'
        self.args.score_threshold = 0.3
        self.args.detect = False
        self.args.yolact_transfer = True
        self.args.trt_batch_size = 4
        self.args.disable_tensorrt = False
        self.args.use_fp16_tensorrt = True
        self.args.no_bar = False
        self.args.display = False
        self.args.resume = False
        self.args.output_coco_json = False
        self.args.output_web_json=False
        self.args.shuffle=False
        self.args.no_hash = False
        self.args.crop = True

        setup_logger(logging_level=logging.INFO)
        self.logger = logging.getLogger('yolact.eval')
        
        self.color_cache = defaultdict(lambda: {})

        with torch.no_grad():
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

            torch2trt_flags = ['torch2trt_backbone', 'torch2trt_backbone_int8', 'torch2trt_protonet', 'torch2trt_protonet_int8', 'torch2trt_fpn', 'torch2trt_fpn_int8', 'torch2trt_prediction_module', 'torch2trt_prediction_module_int8', 'torch2trt_spa', 'torch2trt_spa_int8', 'torch2trt_flow_net', 'torch2trt_flow_net_int8']

            if self.args.use_fp16_tensorrt:
                for key in torch2trt_flags:
                    if 'int8' in key and getattr(cfg, key, False):
                        setattr(cfg, key, False)
                        setattr(cfg, key[:-5], True)

            self.logger.info('Loading model...')
            self.net = Yolact(training=False)
            if self.args.trained_model is not None:
                self.net.load_weights(self.args.trained_model, args=self.args)
            else:
                self.logger.warning('No weights loaded!')
            self.net.eval()
            self.logger.info('Model loaded.')

            use_tensorrt_conversion = any(getattr(cfg, key, False) for key in torch2trt_flags)
            if use_tensorrt_conversion:
                self.logger.info('Converting to TensorRT...')
            
            self.net.model_path = self.args.trained_model

            calibration_dataset = None
            calibration_protonet_dataset = None
            calibration_ph_dataset = None
            calibration_fpn_dataset = None
            calibration_flow_net_dataset = None

############### uncomment if use int8 rather than fp16
            # if (cfg.torch2trt_backbone_int8 or cfg.torch2trt_protonet_int8 or cfg.torch2trt_flow_net_int8):
            #     if (not cfg.torch2trt_backbone_int8 or net.has_trt_cached_module('backbone', True)) and \
            #         (not cfg.torch2trt_protonet_int8 or net.has_trt_cached_module('proto_net', True)) and \
            #             (not cfg.torch2trt_flow_net_int8 or net.has_trt_cached_module('flow_net', True)):
            #         logger.debug('Skipping generation of calibration dataset for backbone/flow_net because there is cache...')
            #     else:
            #         logger.debug('Generating calibration dataset for backbone of {} images...'.format(cfg.torch2trt_max_calibration_images))

            #         calib_images = cfg.dataset.calib_images
            #         if args.calib_images is not None:
            #             calib_images = args.calib_images

            #         def pull_calib_dataset(calib_folder, transform=BaseTransform()):
            #             images = []
            #             for p in Path(calib_folder).glob('*'): 
            #                 path = str(p)
            #                 img = cv2.imread(path)
            #                 height, width, _ = img.shape

            #                 img, _, _, _ = transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
            #                     {'num_crowds': 0, 'labels': np.array([0])})

            #                 images.append(torch.from_numpy(img).permute(2, 0, 1))

            #             calibration_dataset = torch.stack(images)
            #             if args.cuda:
            #                 calibration_dataset = calibration_dataset.cuda()
            #             return calibration_dataset

            #         if ':' in calib_images:
            #             calib_dir, prev_folder, next_folder = calib_images.split(':')
            #             prev_dir = os.path.join(calib_dir, prev_folder)
            #             next_dir = os.path.join(calib_dir, next_folder)

            #             calibration_dataset = pull_calib_dataset(prev_dir)
            #             calibration_next_dataset = pull_calib_dataset(next_dir)
            #         else:
            #             calibration_dataset = pull_calib_dataset(calib_images)

            # n_images_per_batch = 1
            # if cfg.torch2trt_protonet_int8:
            #     if net.has_trt_cached_module('proto_net', True):
            #         logger.debug('Skipping generation of calibration dataset for protonet because there is cache...')
            #     else:
            #         logger.debug('Generating calibration dataset for protonet with {} images...'.format(cfg.torch2trt_max_calibration_images))
            #         calibration_protonet_dataset = []

            #         def forward_hook(self, inputs, outputs):
            #             calibration_protonet_dataset.append(inputs[0])

            #         proto_net_handle = net.proto_net.register_forward_hook(forward_hook)

            # if (cfg.torch2trt_protonet_int8 or cfg.torch2trt_flow_net_int8):
            #     if (not cfg.torch2trt_protonet_int8 or net.has_trt_cached_module('proto_net', True)) and (not cfg.torch2trt_flow_net_int8 or net.has_trt_cached_module('flow_net', True)):
            #         logger.debug('Skipping generation of calibration dataset for protonet/flow_net because there is cache...')
            #     else:
            #         with torch.no_grad():
            #             laterals = []
            #             f1, f2, f3 = [], [], []
            #             for i in range(math.ceil(cfg.torch2trt_max_calibration_images / n_images_per_batch)):
            #                 gt_forward_out = net(calibration_dataset[i*n_images_per_batch:(i+1)*n_images_per_batch], extras={
            #                     "backbone": "full",
            #                     "keep_statistics": True,
            #                     "moving_statistics": None
            #                 })
            #                 laterals.append(gt_forward_out["lateral"])
            #                 f1.append(gt_forward_out["feats"][0])
            #                 f2.append(gt_forward_out["feats"][1])
            #                 f3.append(gt_forward_out["feats"][2])

            #         laterals = torch.cat(laterals, dim=0)
            #         f1 = torch.cat(f1, dim=0)
            #         f2 = torch.cat(f2, dim=0)
            #         f3 = torch.cat(f3, dim=0)

            # if cfg.torch2trt_protonet_int8:
            #     if net.has_trt_cached_module('proto_net', True):
            #         logger.debug('Skipping generation of calibration dataset for protonet because there is cache...')
            #     else:
            #         proto_net_handle.remove()
            #         calibration_protonet_dataset = torch.cat(calibration_protonet_dataset, dim=0)

            # if cfg.torch2trt_flow_net_int8:
            #     if net.has_trt_cached_module('flow_net', True):
            #         logger.debug('Skipping generation of calibration dataset for flow_net because there is cache...')
            #     else:
            #         logger.debug('Generating calibration dataset for flow_net with {} images...'.format(cfg.torch2trt_max_calibration_images))
            #         calibration_flow_net_dataset = []

            #         def forward_hook(self, inputs, outputs):
            #             calibration_flow_net_dataset.append(inputs[0])

            #         handle = net.flow_net.flow_net.register_forward_hook(forward_hook)
            #         for i in range(math.ceil(cfg.torch2trt_max_calibration_images / n_images_per_batch)):
            #             extras = {
            #                 "backbone": "partial",
            #                 "moving_statistics": {
            #                     "lateral": laterals[i*n_images_per_batch:(i+1)*n_images_per_batch],
            #                     "feats": [
            #                         f1[i*n_images_per_batch:(i+1)*n_images_per_batch],
            #                         f2[i*n_images_per_batch:(i+1)*n_images_per_batch],
            #                         f3[i*n_images_per_batch:(i+1)*n_images_per_batch]
            #                     ]
            #                 }
            #             }
            #             with torch.no_grad():
            #                 net(calibration_next_dataset[i*n_images_per_batch:(i+1)*n_images_per_batch], extras=extras)
            #         handle.remove()

            #         calibration_flow_net_dataset = torch.cat(calibration_flow_net_dataset, dim=0)


            if cfg.torch2trt_backbone or cfg.torch2trt_backbone_int8:
                self.logger.info("Converting backbone to TensorRT...")
                self.net.to_tensorrt_backbone(cfg.torch2trt_backbone_int8, calibration_dataset=calibration_dataset, batch_size=self.args.trt_batch_size)

            if cfg.torch2trt_protonet or cfg.torch2trt_protonet_int8:
                self.logger.info("Converting protonet to TensorRT...")
                self.net.to_tensorrt_protonet(cfg.torch2trt_protonet_int8, calibration_dataset=calibration_protonet_dataset, batch_size=self.args.trt_batch_size)

            if cfg.torch2trt_fpn or cfg.torch2trt_fpn_int8:
                self.logger.info("Converting FPN to TensorRT...")
                self.net.to_tensorrt_fpn(cfg.torch2trt_fpn_int8, batch_size=self.args.trt_batch_size)
                # net.fpn_phase_1.to_tensorrt(cfg.torch2trt_fpn_int8)
                # net.fpn_phase_2.to_tensorrt(cfg.torch2trt_fpn_int8)

            if cfg.torch2trt_prediction_module or cfg.torch2trt_prediction_module_int8:
                self.logger.info("Converting PredictionModule to TensorRT...")
                self.net.to_tensorrt_prediction_head(cfg.torch2trt_prediction_module_int8, batch_size=self.args.trt_batch_size)
                # for prediction_layer in net.prediction_layers:
                #     prediction_layer.to_tensorrt(cfg.torch2trt_prediction_module_int8)

            if cfg.torch2trt_spa or cfg.torch2trt_spa_int8:
                self.logger.info('Converting SPA to TensorRT...')
                assert not cfg.torch2trt_spa_int8
                self.net.to_tensorrt_spa(cfg.torch2trt_spa_int8, batch_size=self.args.trt_batch_size)

            if cfg.torch2trt_flow_net or cfg.torch2trt_flow_net_int8:
                self.logger.info('Converting flow_net to TensorRT...')
                self.net.to_tensorrt_flow_net(cfg.torch2trt_flow_net_int8, calibration_dataset=calibration_flow_net_dataset, batch_size=self.args.trt_batch_size)

            if use_tensorrt_conversion:
                self.logger.info("Converted to TensorRT.")

    def detect(self, img):
        with torch.no_grad():
            self.net = self.net.cuda()
            self.net.detect.use_fast_nms = self.args.fast_nms
            cfg.mask_proto_debug = self.args.mask_proto_debug
            img_out = self.evalimage(img)
            return img_out

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

    def evalimage(self, img, save_path=None):
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))

        if cfg.flow.warp_mode != 'none':
            assert False, 'Evaluating the image with a video-based model.'

        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}

        start_time = time.time()
        preds = self.net(batch, extras=extras)["pred_outs"]
        end_time = time.time()
        print('%.3f' % (end_time-start_time))

        img_numpy = self.prep_display(preds, frame, None, None, undo_transform=False)
        if save_path:
            cv2.imwrite(save_path, img_numpy)
        return img_numpy

    def prep_display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            t = postprocess(dets_out, w, h, visualize_lincomb = self.args.display_linecomb,
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

        if num_dets_to_consider == 0:
            # no detection found so just output original image
            return (img_gpu * 255).byte().cpu().numpy()

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
            masks = masks[:num_dets_to_consider, :, :, None]

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
        return img_numpy

#if __name__ == '__main__':
#    engine = YolactEdgeEngine()
#    engine.evaluate()

