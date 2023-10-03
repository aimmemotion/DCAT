# Most Important Person-guided Dual-branch Cross-Patch Attention for Group Affect Recognition (ICCV 2023)
This repository is the official implementation of "[Most Important Person-guided Dual-branch Cross-Patch Attention for Group Affect Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Xie_Most_Important_Person-Guided_Dual-Branch_Cross-Patch_Attention_for_Group_Affect_Recognition_ICCV_2023_paper.html)". Please feel free to reach out for any questions or discussions.

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{xie2023most,
  title={Most Important Person-Guided Dual-Branch Cross-Patch Attention for Group Affect Recognition},
  author={Xie, Hongxia and Lee, Ming-Xian and Chen, Tzu-Jui and Chen, Hung-Jen and Liu, Hou-I and Shuai, Hong-Han and Cheng, Wen-Huang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20598--20608},
  year={2023}
}
```

### Abstract
Group affect refers to the subjective emotion that is evoked by an external stimulus in a group, which is an important factor that shapes group behavior and outcomes. Recognizing group affect involves identifying important individuals and salient objects among a crowd that can evoke emotions. However, most existing methods lack attention to affective meaning in group dynamics and fail to account for the contextual relevance of faces and objects in group-level images. In this work, we propose a solution by incorporating the psychological concept of the Most Important Person (MIP), which represents the most noteworthy face in a crowd and has affective semantic meaning. We present the Dual-branch Cross-Patch Attention Transformer (DCAT) which uses global image and MIP together as inputs. Specifically, we first learn the informative facial regions produced by the MIP and the global context separately. Then, the Cross-Patch Attention module is proposed to fuse the features of MIP and global context together to complement each other. Our proposed method outperforms state-of-the-art methods on GAF 3.0, GroupEmoW, and HECO datasets. Moreover, we demonstrate the potential for broader applications by showing that our proposed model can be transferred to another group affect task, group cohesion, and achieve comparable results.
