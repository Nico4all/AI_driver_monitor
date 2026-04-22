###COMANDOS PARA USAR CONTENEDOR

## REQUERIMIENTOS
Tener activo la venv de docker con cuda y drivers nvidia actualizados
Tener instalado Nvidia Container Toolkit

##Comandos

# 1. Build
docker build -t driver_monitor .

# 2. Entrar al contenedor
docker run --gpus all -it -v "${PWD}:/app" driver_monitor bash

# 3. Entrenar Mask R-CNN (~30 min)
cd /app/src
python train_maskrcnn.py

# 4. Entrenar clasificador con 5-fold CV (~15 min)
python train_classifier.py

# 5. Evaluar ambos
python evaluate.py