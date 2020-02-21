This is learning project to understand how machine learning works.
Aim - create and tech network to do text on picture recognition.

Whole process split by few steps:
1) We should grab/download some images to use them as train/validate set, please use ```grabber.py``` as an entry point, inside file you can tweak how much data do you need
2) Your downloaded pictures should be described in the ```desc.json``` where you put picture name and exact text it contains
3) Run ```image_transform.py``` to do image normalization, it will remove picture noise, convert to black/white and crop to appropriate size
4) Run ```neuro_image.py``` to actually start teaching your network (pay attention, currently network storing commented out)
5) Run ```model.py``` to load pre-trained network and try to recognize real picture

Didn't try to make it as a finally solution, so all files probably requires minimal tweaks.