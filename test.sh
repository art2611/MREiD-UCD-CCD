#!/bin/sh

echo " "
echo "Enter number X to select option X:"
echo " "

#number=12345
#digit_list=()
#
## Convert the number to a string
#number_string="$number"
#
## Iterate over each character in the string
#for (( i=0; i<${#number_string}; i++ )); do
#    digit="${number_string:i:1}"
#    digit_list+=("$digit")  # Add the digit to the list
#done

echo "Fast run ? enter FCODE:"


MODELS_PATH="../save_model_contour_bimodal"
DATA_PATH="../Datasets"
echo "Pre-configured MODELS_PATH: ${MODELS_PATH}"
echo "Pre-configured DATA_PATH: ${DATA_PATH}"
echo " "

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

read -e -p "Scenario evaluation - (1) normal - (2) C - (3) UCD - (4) CCD :" SCENARIO_EVAL
Scenario_eval=("normal" "C" "UCD" "CCD")
SCENARIO_EVAL=${Scenario_eval[SCENARIO_EVAL-1]}
echo $SCENARIO_EVAL

CCDX=0
if [ ${SCENARIO_EVAL} == "CCD" ]
then
      read -e -p "CCD-X value ? Default=0, max=100:" CCDX
      echo "CCD-${CCDX}"
fi

read -e -p "GPU value ? (0) - (1) - (2) - ... :" GPU
export CUDA_VISIBLE_DEVICES=$GPU
echo $CUDA_VISIBLE_DEVICES

echo

python test.py   --model=$MODEL \
                  --dataset=$DATASET \
                  --models_path=$MODELS_PATH \
                  --data_path=$DATA_PATH \
                  --scenario_eval=$SCENARIO_EVAL \
                  --reid=$REID \
                  --CCD_X=$CCDX \
                  $MLMDA \
                  $CIL;

