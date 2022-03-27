################################################################################
#                PRIMA PARTE: Rimozione Background (detectron2)
################################################################################

# Cercare un modo per importare il set allenato nel file "visualizer.py",
# altrimenti rifaccio lo stesso allenamento con il dataset "Tango".

################################################################################
#                     SECONDA PARTE: Generazione Modello 3D
################################################################################

# Inserire il modello che usa PyTorch3D di Marco per la generazione del tango3D
# tramite singola view del Tango

################################################################################
#                  TERZA PARTE: Allenamento riconoscimento pose
################################################################################

# Usare la procedura del tutorial su PyTorch3D
# (https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb)
# adattandola ai dati necessari per la submission

################################################################################
#                    QUARTA PARTE: Ottenimento pose del test set
################################################################################

# Classico eval sul modello usando la parte di test del dataset

################################################################################
#                      QUINTA PARTE: Preparazione Submission
################################################################################

# Usare il codice fornito da esa (su GitLab) per la generazione del JSON per la
# subminssion da valutare
