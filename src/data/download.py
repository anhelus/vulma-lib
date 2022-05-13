from roboflow import Roboflow
rf = Roboflow(api_key="37us8AlkgbjNCKzlVNTz")
project = rf.workspace("angelo-cardellicchio").project("bridge-vulma")
dataset = project.version(2).download("yolov5")