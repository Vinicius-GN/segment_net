import numpy as np
from ast import literal_eval as make_tuple


def split_config(args):

	config = dict()

	# dirs 
	config["dirs"] = {
			"data": args.get("DIRS").get("data_folder"),
			"logs":args.get("DIRS").get("logs"),
			"metadata":args.get("DIRS").get("metadata_path"),
            "mapping":args.get("DIRS").get("mapping_path")
		 }	

	# splits
	if args.get("DATA").get("dataset") == "rellis3d" or\
       args.get("DATA").get("dataset") == "goose" or\
       args.get("DATA").get("dataset") == "bdd100k":
		config["splits"] = {
				"train": args.get("SPLITS").get("train_seqs"),
				"val":   args.get("SPLITS").get("val_seqs"),
				"test":  args.get("SPLITS").get("test_sets"),
			}
	else:
		config["splits"] = {
				"train": make_tuple(args.get("SPLITS").get("train_seqs")),
				"val":   make_tuple(args.get("SPLITS").get("val_seqs")),
				"test":  make_tuple(args.get("SPLITS").get("test_sets")),
			}

	# data    									
	config["data"] = {
			"batch_size": int(args.get("DATA").get("batch_size")),
			"num_samples_plot": int(args.get("DATA").get("num_samples_plot")),
			"num_samples": int(args.get("DATA").get("num_samples")),
			"dataset": args.get("DATA").get("dataset"),
		}

	# train
	config["train"] = {
			"epochs":int(args.get("TRAIN").get("epochs")),
            "patience":int(args.get("TRAIN").get("patience")),
			"steps_minibatch_optimizer":int(args.get("TRAIN").get("steps_minibatch_optimizer"))
		}
	
	# checkpoint
	config["checkpoint"] = {
			"steps_train" : int(args.get("CHECKPOINT").get("steps_train")),
			"save_best": args.get("CHECKPOINT").get("save_best") == "true",
			"save_last": args.get("CHECKPOINT").get("save_last") == "true",
			"steps_plot_train":  int(args.get("CHECKPOINT").get("steps_plot_train"))
		}

	# model
	config["model"] = {
          "learning_rate" : float(args.get("MODEL").get("learning_rate")),
          "weight_decay"  : float(args.get("MODEL").get("weight_decay")),
          "dropout": float(args.get("MODEL").get("dropout")),
		  "grad_norm": float(args.get("MODEL").get("grad_norm")),
		  "norm_fn": args.get("MODEL").get("norm_fn"),
	}
	
	# backbone
	config["backbone"] = {
		"type": args.get("BACKBONE").get("type"),
  		"fpn_out_channels": int(args.get("BACKBONE").get("fpn_out_channels")),
        "aggregate":args.get("BACKBONE").get("aggregate"),
	    "layer_weights":make_tuple(args.get("BACKBONE").get("layer_weights")),
        "dropout": float(args.get("BACKBONE").get("dropout")),
	}
 
	# head
	config["head"] = {
    	  "dropout": float(args.get("HEAD").get("dropout")),		  
    	  "type":args.get("HEAD").get("type"),
	      "num_blocks": int(args.get("HEAD").get("num_blocks")),
          "opt_latency": args.get("HEAD").get("opt_latency") == "true",
          "num_heads_transformer": int(args.get("HEAD").get("num_head_transformer")),
	}
	# attention
	config["attention"] = {
		"reduction_rate": int(args.get("ATTENTION").get("reduction_rate")),
		"use_attention" : args.get("ATTENTION").get("type")!="none",
		"type": args.get("ATTENTION").get("type"),
		"dropout":float(args.get("ATTENTION").get("dropout"))
	}
 
    # mode
	config["mode"] = {
		"mode" : args.get("MODE").get("mode")
	}
     
	# loss
	config["loss"] = {
		"type" : args.get("LOSS").get("type"),
		"alpha": float(args.get("LOSS").get("alpha")),
		"gamma": float(args.get("LOSS").get("gamma")),
		"class_weights":make_tuple(args.get("LOSS").get("class_weights")),
		"loss_scale": float(args.get("LOSS").get("loss_scale"))
	}

	# image
	config["image"]={
		"num_classes": int(args.get("IMAGE").get("num_classes")),       
        "image_size":make_tuple(args.get("IMAGE").get("image_size")),
        "resize":args.get("IMAGE").get("resize") == "true",
	}
    
	return config


def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def color_image(image, mapping):
	
	H, W = image.shape

	colored_image = np.zeros((H, W, 3))

	for c, v in mapping.items():
		index = np.where(image==c)
		colored_image[index[0], index[1], :] = v
	
	return colored_image

def color_pc(pc, mapping):

	print(pc.shape)
	N = pc.shape[0]

	colored_points = np.zeros((N, 3))

	for c, v in mapping.items():
		index = np.where(pc==c)[0]
		print(index)
		colored_points[index] = v
	
	return colored_points

def to_serializable(obj):
    if isinstance(obj, (np.integer, np.uint64, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj