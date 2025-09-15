#!/bin/bash

configs=(
    "/home/lrm/workspace/segment_net/cfg/goose_4090_maxxvitv2_deeplabhead.ini"
    "/home/lrm/workspace/segment_net/cfg/goose_4090_maxxvitv2_seghead.ini"
)

#Backbone
#  "cfg/goose_4090_segformerb2.ini"
#  "cfg/goose_4090_segformerb0.ini"
#  "cfg/goose_4090_swinv2.ini"
#  "cfg/goose_4090_edgenext.ini"
#  "cfg/goose_4090_efficientformer.ini"
#  "cfg/goose_4090_fastvit.ini" """
#  "cfg/goose_4090_mobilevit.ini"
#  "cfg/goose_4090_pitxs.ini"
#  "cfg/goose_4090_sam2_hiera.ini"
#  "cfg/goose_4090_tinyvit.ini"

#Mecanismos de atenção
#cfg/experiments_bestnet/goose_4090_maxxvit_classchannel.ini 
#cfg/experiments_bestnet/goose_4090_maxxvit_none.ini 
#cfg/experiments_bestnet/goose_4090_maxxvit_query.ini 
#cfg/experiments_bestnet/goose_4090_maxxvit_sechannel.ini 
#cfg/experiments_bestnet/goose_4090_maxxvit_spatial.ini

#Optimizer
#/home/lrm/workspace/segment_net/cfg/experiments_bestnet/goose_4090_maxxvit_spatial_withoutopt.ini

#Decoder Head:
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/Head/goose_4090_maxxvit_query_with_conv_interp.ini"
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/Head/goose_4090_maxxvit_query_with_depthwise_nn.ini"

#Aggreagtion Operations:
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/Operations/goose_4090_maxxvit_spatial_concat.ini"
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/Operations/goose_4090_maxxvit_spatial_sum.ini"
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/Operations/goose_4090_maxxvit_spatial_wsum.ini"

# FPN:
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/fpn/goose_4090_maxxvit_spatial_64.ini"
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/fpn/goose_4090_maxxvit_spatial_128.ini"
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/fpn/goose_4090_maxxvit_spatial_256.ini"
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/fpn/goose_4090_maxxvit_spatial_512.ini"
#"/home/lrm/workspace/segment_net/cfg/experiments_bestnet/fpn/goose_4090_maxxvit_spatial1024.ini"

#New heads:
#    "/home/lrm/workspace/segment_net/cfg/goose_4090_segformerb2_seghead.ini"
#    "/home/lrm/workspace/segment_net/cfg/goose_4090_segformerb2_deeplabhead.ini"
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
