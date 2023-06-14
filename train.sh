#!/bin/sh

echo " "
echo "Enter number X to select option X:"
echo " "

MODELS_PATH="../save_model_contour_bimodal"
MODELS_PATH="../save_model"
DATA_PATH="../Datasets"
echo "Pre-configured MODELS_PATH: ${MODELS_PATH}"
echo "Pre-configured DATA_PATH: ${DATA_PATH}"

read -e -p "Dataset : (1) SYSU - (2) RegDB - (3) TWorld :" DATASET
Datasets=("SYSU" "RegDB" "TWorld")
DATASET=${Datasets[DATASET-1]}
echo $DATASET

read -e -p "Model : (1) concatenation - (2) unimodal - (3) transreid - (4) LightMBN (5) MMSF:" MODEL
Model=("concatenation" "unimodal" "transreid" "LMBN" "MMSF")
MODEL=${Model[MODEL-1]}
echo $MODEL

if [ "$MODEL" == "MMSF" ]
then
  read -e -p "Model : (0) MMSF_0 - (1) MMSF_1 ... - (4) MMSF_4 :" MODEL
Model=("MMSF_0" "MMSF_1" "MMSF_2" "MMSF_3" "MMSF_4")
MODEL=${Model[MODEL]}
echo $MODEL
fi

for item in "unimodal" "transreid" "LightMBN"
do
    if [ "$MODEL" == "$item" ]; then
      read -e -p "Train using CIL ? (1) True - (2) False:" CIL
      Cil=("--CIL" " ")
      CIL=${Cil[CIL-1]}
      echo $CIL

      REID="VtoV" # VtoV for all unimodal models
  fi
done

if [ -z ${CIL} ] | [[ ${CIL} != " " &&  ${CIL} != "--CIL" ]]
then
      read -e -p "Train using ML-MDA ? (1) True - (2) False:" MLMDA
      Mlmda=("--ML_MDA" " ")
      MLMDA=${Mlmda[MLMDA-1]}
      echo $MLMDA

      REID="BtoB" # BtoB for multimodal model
fi

# Select GPU
read -e -p "GPU value ? (0) - (1) - (2) - ... :" GPU
export CUDA_VISIBLE_DEVICES=$GPU
echo $CUDA_VISIBLE_DEVICES

echo

# Train the 5 folds direcly
for fold in `seq 0 4`;
  do
    python train.py   --model=$MODEL \
                      --dataset=$DATASET \
                      --models_path=$MODELS_PATH \
                      --data_path=$DATA_PATH \
                      --reid=$REID \
                      --fold=$fold \
                      $MLMDA \
                      $CIL;
done