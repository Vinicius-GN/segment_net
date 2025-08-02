#!/bin/bash
configs=( 
    "cfg/rellis3d_5090_pitxs.ini"
    "cfg/rellis3d_5090_levit.ini"
    "cfg/rellis3d_5090_effiicientformer.ini"  
)

#"cfg/rellis3d_5090_mobilevit.ini"
#"cfg/rellis3d_5090_segformerb0.ini"
#"cfg/rellis3d_5090_sam2_hiera.ini"

#Retreinar
#"cfg/rellis3d_5090_effiicientformer.ini"   
#"cfg/rellis3d_5090_levit.ini"
#"cfg/rellis3d_5090_deit3_small.ini"

#Testar:
#"cfg/rellis3d_5090_convnextv2.ini"
#"cfg/rellis3d_5090_tinyvit.ini"
#"cfg/rellis3d_5090_fastvit.ini"



# Loop para rodar cada configuração
for cfg in "${configs[@]}"; do
    echo "Rodando treinamento com $cfg"
    python run.py --cfg "$cfg"

    # Verifica se houve erro no último comando
    if [ $? -ne 0 ]; then
        echo "Erro ao rodar $cfg. Interrompendo a sequência."
        exit 1
    fi

    sleep 120
done

echo "Todos os treinamentos foram concluídos"
