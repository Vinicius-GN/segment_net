#!/bin/bash

configs=( 
"cfg/rellis3d_5090_convnextv2.ini"
"cfg/rellis3d_5090_deeplab.ini"
)

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
