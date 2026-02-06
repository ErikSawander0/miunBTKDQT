"""
file: ./Student.py
Helper class, creates students
"""
from transformers import VitPoseForPoseEstimation, VitPoseConfig, VitPoseBackboneConfig

# we want to set this up so that it can clone weights at specific depths...
def createStudent(depth, layer_mapping, teacher_sd ): 

    if depth != len(layer_mapping) or len(layer_mapping) > 12:
        raise ValueError(f"bad layer mapping on depth: {depth}")
    backbone_config = VitPoseBackboneConfig(
        num_hidden_layers=depth,
        hidden_size=768,
        num_attention_heads=12,
        initializer_range= 0.02,
        layer_norm_eps= 1e-12,
        mlp_ratio= 4,
        model_type= "vitpose_backbone",
        num_channels= 3,
        patch_size= [
          16,
          16
        ],
        qkv_bias= True,

        scale_factor= 4,
        use_pretrained_backbone= False,
        use_simple_decoder= True,
        use_timm_backbone= False
    )

    config = VitPoseConfig(
        backbone_config=backbone_config,
        num_labels=17,
    )
    student = VitPoseForPoseEstimation(config)

    student_sd = student.state_dict()
    for student_idx, teacher_idx in layer_mapping.items():
        if student_idx < 0 or student_idx >= depth:
            raise ValueError(f"Student layer {student_idx} doesn't exist (model has {depth} layers)")
        if teacher_idx < 0 or teacher_idx >= 12:
            raise ValueError(f"Teacher layer {teacher_idx} doesn't exist (model has {depth} layers)")

        for key in student_sd.keys():
            if f".layer.{student_idx}." in key:
                teacher_key = key.replace(f".layer.{student_idx}.", f".layer.{teacher_idx}.")
                student_sd[key] = teacher_sd[teacher_key]
    student.load_state_dict(student_sd)
    return student

