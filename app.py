
import streamlit as st
import streamlit.components.v1 as stc

# File Processing Pkgs
import pandas as pd
from PIL import Image 

import numpy as np
import cv2
import torch

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# import fitz  # this is pymupdf

# def read_pdf_with_fitz(file):
# 	with fitz.open(file) as doc:
# 		text = ""
# 		for page in doc:
# 			text += page.getText()
# 		return text 

# Fxn
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 



def main():
	st.title("File Upload Tutorial")

	menu = ["Home","Dataset","DocumentFiles","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	print("detectron2 imported")
	st.info("start keypoints:")

	if choice == "Home":
		st.subheader("Home")
		image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
		if image_file is not None:
		
			# To See Details
			# st.write(type(image_file))
			# st.write(dir(image_file))
			file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
			st.write(file_details)

			img = load_image(image_file)
			st.image(img,width=250)
			config_file_path = "models/config.yml"
			weights_path = "models/model_final.pth"

			image_path = "fencing.jpg"
			keypoints = "kyepoint" #make_inference(image_path, config_file_path, weights_path, threshold=0.90) #getKeypointsFromPredictor("image_path",st)
			st.info("keypoints:"+keypoints)
			keypoints = make_inference(image_path, config_file_path, weights_path, threshold=0.90) #getKeypointsFromPredictor("image_path",st)
			st.info(keypoints)

	elif choice == "Dataset":
		st.subheader("Dataset")
		data_file = st.file_uploader("Upload CSV",type=['csv'])
		if st.button("Process"):
			if data_file is not None:
				file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
				st.write(file_details)

				df = pd.read_csv(data_file)
				st.dataframe(df)

	elif choice == "DocumentFiles":
		st.subheader("DocumentFiles")
		docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
		if st.button("Process"):
			if docx_file is not None:
				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
				st.write(file_details)
				# Check File Type
				if docx_file.type == "text/plain":
					# raw_text = docx_file.read() # read as bytes
					# st.write(raw_text)
					# st.text(raw_text) # fails
					st.text(str(docx_file.read(),"utf-8")) # empty
					raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
					# st.text(raw_text) # Works
					st.write(raw_text) # works

	else:
		st.subheader("About")
		st.info("Built with Streamlit")
		st.info("Jesus Saves @JCharisTech")
		st.text("Jesse E.Agbe(JCharis)")

def getKeypointsFromPredictor(image_path, st):
    print("getKeypointsFromPredictor")
    st.info("getKeypointsFromPredictor")
    config_file_path = "models/config.yml"

    weights_path = "models/model_final.pth"

    image_path = "fencing.jpg"

    model = config_file_path
    im = cv2.imread(image_path)
    st.info("got image")
    cfg = get_cfg()
    cfg.merge_from_file(config_file_path)
    st.info("got config_file")
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
    st.info("got weights_path")
    predictor = DefaultPredictor(cfg)
    st.info("got predictor")
    outputs = predictor(im)
    st.info("got outputs")
    keypoints = outputs["instances"].pred_keypoints
    print("keypoints: ", keypoints)
    st.info("end getKeypointsFromPredictor")
    return keypoints

#@st.cache(allow_output_mutation=True)
def create_predictor(model_config, model_weights, threshold):
    """
    Loads a Detectron2 model based on model_config, model_weights and creates a default
    Detectron2 predictor.
    Returns Detectron2 default predictor and model config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_config)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(torch.cuda.is_available())
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.SCORE_THRESH_TEST = threshold
    st.info("before predictor")
    print("before predictor")
    predictor = DefaultPredictor(cfg)
    st.info("got predictor")
    print("got predictor")
    return cfg, predictor

def make_inference(image_path, model_config, model_weights, threshold=0.5, n=5, save=False):
  """
  Makes inference on image (single image) using model_config, model_weights and threshold.
  Returns image with n instance predictions drawn on.
  Params:
  -------
  image (str) : file path to target image
  model_config (str) : file path to model config in .yaml format
  model_weights (str) : file path to model weights 
  threshold (float) : confidence threshold for model prediction, default 0.5
  n (int) : number of prediction instances to draw on, default 5
    Note: some images may not have 5 instances to draw on depending on threshold,
    n=5 means the top 5 instances above the threshold will be drawn on.
  save (bool) : if True will save image with predicted instances to file, default False
  """
  # Create predictor and model config
  st.info("in make_inference")
  cfg, predictor = create_predictor(model_config, model_weights, threshold)
  im = cv2.imread(image_path)
  st.info("got image")
  outputs = predictor(im)
  st.info("got outputs")
  keypoints = outputs["instances"].pred_keypoints
  print("keypoints: ", keypoints)
  st.info("end getKeypointsFromPredictor")
  return keypoints

if __name__ == '__main__':
	main()