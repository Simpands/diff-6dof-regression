import os

print("CANCELLAZIONE VECCHI FILE")
os.system("rm -r /data/simpsi/train/images")
os.system("rm /data/simpsi/train/annotation.json")
print("RICREO CARTELLA VUOTA PER IMMAGINI DATASET")
os.system("mkdir /data/simpsi/train/images")
print("FINE")