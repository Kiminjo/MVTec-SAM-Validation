# MVTec-SAM-Validation

This repository aims to measure the zero-shot segmentation performance of Segment Anything Models (SAM) on Industrial defect data 
<br></br>

## Experiment configs 
- prompt type
    - bounding box 
    - point 
- data 
    - MVTec bottle 
    - MVTec screw 
    - MVTec wood data 

<br>

## RUN 
1. **Get data using dvc**
```
cd data 
dvc pull mvtec.dvc 
```
<br>

2. **Set MVTec's category in `main.py`**
```
# main.py line 11
category = "wood" # Change here 
```
<br>

3. **Run Code**
```
python main.py
```

Then the `results` folder will be created and contain the inference result images.
<br>

4. **(Option) Load previously learned results**
```
dvc pull results.dvc 
```