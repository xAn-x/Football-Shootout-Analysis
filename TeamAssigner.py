from transformers import AutoProcessor,AutoModel
import torch
from toolz.itertoolz import partition_all
from tqdm import tqdm
import pickle
import numpy as np
import supervision as sv
import cv2

class TeamAssigner():
  def __init__(self,EMBEDDING_MODEL:str="google/siglip-base-patch16-224",REDUCER:str=rf"./weights/REDUCER.pkl",CLUSTERING_MODEL:str=rf"./weights/CLUSTERING_MODEL.pkl") -> None:

    self.EMBEDDING_PROCESSOR=AutoProcessor.from_pretrained(EMBEDDING_MODEL,cache_dir=rf"./weights")
    self.EMBEDDING_MODEL=AutoModel.from_pretrained(EMBEDDING_MODEL,cache_dir=rf"./weights")


    with open(str(REDUCER), 'rb') as f:
      self.REDUCER = pickle.load(f)

    with open(str(CLUSTERING_MODEL), 'rb') as f:
      self.CLUSTERING_MODEL = pickle.load(f)

    self.DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

  @torch.no_grad()
  def get_team_ids(self,frame,detections:sv.Detections,batch_size=16,verbose=True) -> np.ndarray:
    if detections.is_empty():
      return np.array([])

    # Extract player crops
    player_crops = []
    temp_frame=frame.copy()
    for detection in detections:
      crop=sv.crop_image(temp_frame,detection[0])
      player_crops.append(crop)

    # convert cv2 to pillow 
    player_crops = [sv.cv2_to_pillow(crop) for crop in player_crops]
  
    # Create Embeddings
    images=partition_all(batch_size,player_crops)

    team_ids=[]
    for batch in tqdm(images,desc="Creating embeddings from crops and generating team-ids",disable=not verbose,total=len(player_crops)//batch_size+1):
      inputs=self.EMBEDDING_PROCESSOR(images=batch,return_tensors="pt").to(self.DEVICE)
      embeddings=self.EMBEDDING_MODEL.get_image_features(**inputs).cpu().numpy()

      # Dimensionality-Reduction: For most valuable and enriched embeddings
      embeddings=self.REDUCER.transform(embeddings)

      # Cluster Embeddings
      team_ids.append(self.CLUSTERING_MODEL.predict(embeddings))
    
    team_ids=np.concatenate(team_ids)

    return team_ids
