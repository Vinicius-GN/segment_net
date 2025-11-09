#!/bin/bash

configs=(
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_DANet.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_Efficientvit.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_OCR.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_PSA_par.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_PSAseq.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_PVT-Linear.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_PVT-SRA.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_query.ini"
    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_spatial.ini"

    #"cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_none.ini"
    "cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_class_channel.ini"
    "cfg/CROS_Experiments/Segformerb2/goose_4090_segformerb2_deeplabhead_se_channel.ini"
 
    #"cfg/dennis/bdd100k_5090_maxxvitv2_ss.ini"
    #"cfg/dennis/bdd100k_5090_maxxvitv2_lovasz.ini"
    #"cfg/dennis/bdd100k_5090_maxxvitv2_focalce.ini"
    #"cfg/dennis/bdd100k_2080_maxxvitv2_topk.ini"
    #"cfg/dennis/bdd100k_2080_maxxvitv2_iou.ini"
    #"cfg/dennis/bdd100k_2080_maxxvitv2_focaltversky.ini"
    #"cfg/dennis/bdd100k_1080_maxxvitv2_logcoshdice.ini"
    #"cfg/dennis/bdd100k_1080_maxxvitv2_dice.ini"   
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
#   "" "/home/lrm/workspace/segment_net/cfg/goose_4090_maxxvitv2_deeplabhead.ini""

#New losses:
#cfg/experiments_bestnet/losses/goose_4090_segformerb2_seghead_bounday.ini"
#"cfg/experiments_bestnet/losses/goose_4090_segformerb2_seghead_dmce.ini"

#New losses and unique vegetation class:    
#"cfg/IV_experiments/losses/goose_4090_segformerb2_seghead_dmce.ini" 
#"cfg/IV_experiments/losses/goose_4090_segformerb2_seghead_boundary.ini" 
#"cfg/IV_experiments/losses/goose_4090_segformerb2_seghead_dice.ini" 
#"cfg/IV_experiments/losses/goose_4090_segformerb2_seghead_inverseform.ini"
#"cfg/IV_experiments/losses/goose_4090_segformerb2_seghead_activeBL.ini" 
#"cfg/IV_experiments/losses/goose_4090_segformerb2_seghead_conditionalBL.ini"  Reimplementar isso aqui

#New attention Mecanisms:

#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_OCR.ini" 
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PSA_par.ini" 
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PSAseq.ini" 
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PVT-Linear.ini" 
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PVT-SRA.ini" 
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_query.ini" 
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_spatial.ini"

#New Experiments:
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_spatial.ini"
#"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_Efficientvit.ini"
#cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_Efficientvit2.ini"
#Outreos experimentos: Usar segformer_h // aumentar tamanho da fpn_read // testar outras operçaões de agregaão em casos de mts features

#Comparison between attention machanisms
   #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_DANet.ini" 
    #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_OCR.ini" 
    #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PSA_par.ini" 
    #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PSAseq.ini" 
    #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PVT-Linear.ini" 
    #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_PVT-SRA.ini" 
    #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_query.ini" 
    #"cfg/IV_experiments/attention/goose_4090_segformerb2_deeplabhead_spatial.ini"
    #cfg/IV_experiments/attention/goose_4090_segformerb2_segformer_h.ini

#Comparison between best nets
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_512_concat.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_deeplab_h_256_max_pool.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_deeplab_h_512_concat.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_bifpn.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_concat.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_max_pool.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_deeplab_h_256_max_pool_none.ini" 


#Maxxvitv2
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_maxxvitv2_segformer_h_512_concat.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_maxxvitv2_segformer_h_256_concat.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_maxxvitv2_segformer_h_256_max_pool.ini" 

#New tries:
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_max_pool_psa_seq.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_efficientvit_dmce.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_efficientvit_hrnet.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_efficientvit_inverseform.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_transformer_256_max_pool.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_1024_max_pool.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_max_pool_spatial.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_max_pool_query.ini"

 #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb5_segformer_h_256_concat.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb5_segformer_h_512_concat.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb5_segformer_h_1024_concat.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb3_segformer_h_512_concat.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb4_segformer_h_512_concat.ini"
    #"cfg/IV_experiments/New_heavy_tests/1024/goose_4090_segformerb2_segformer_h_1024_concat.ini" ------> HAS BEATEN THE VALUE
    #"cfg/IV_experiments/New_heavy_tests/1024/goose_4090_segformerb2_deeplabv3_h_1024_concat.ini" 

    
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb4_segformer_h_256_concat.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb4_segformer_h_256_max_pool.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb3_segformer_h_256_concat.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb3_segformer_h_256_max_pool.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_deeplabv3_h_1024_max_pool.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_deeplab_h_256_concat.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_deeplab_h_256_max_pool.ini"
    
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_1024_max_pool.ini"
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_concat.ini" 
    #"cfg/IV_experiments/New_heavy_tests/goose_4090_segformerb2_segformer_h_256_max_pool.ini" 

    

for cfg in "${configs[@]}"; do
    echo "Running training with $cfg"
    python run.py --cfg "$cfg"

    if [ $? -ne 0 ]; then
        echo "Error while running $cfg. Interrruptiong the sequence."
        exit 1
    fi

    sleep 120
done

echo "All trainings were concluded"
