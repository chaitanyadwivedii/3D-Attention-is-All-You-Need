# 3D-Attention-is-All-You-Need
A transformer based video question answering system 


## INTRODUCTION
Understanding visual contents, at human-level is the holy grail in visual intelligence. To this end,there has been a tremendous amount of research to solve visual question answering. However, thismostly involves image input, and the video domain still is still nascent. Among other reasons, the
additional temporal aspect makes this a challenging task.
This project makes use of two baseline architectures: LXMERT and Slow Fast networks to perform video question answering. 

## DATA
Our work utilizes the FrameQA category subset from the TGIF-QA dataset. The TGIF-QA dataset
itself is built on top of the the TGIF dataset. It contains 100K animated GIFs and 120K sentences
describing visual content of the animated GIFs. The TGIF-QA dataset contains 165K QA pairs for
the animated GIFs from TGIF. There are four categories of questions - Repeating Action, Repetition
Count, State Transition and Frame QA. 

## METHOD
For tasks of visual-and-language reasoning, LXMERT uses a large scale Transformer model that
is primarily composed of three encoders: an object-relationship encoder, a language encoder, and a
cross-modality encoder. These encoders were pre-trained on several vision and language model tasks,
along with tasks that involved both visual and language aspects like cross-modality matching and
image question-answering.
An image and a related sentence is sent as input to LXMERT, which are converted into embeddings
before sending to the single-modality encoders. Output of both these encoders(Object-Relation and
Language) is fed to the ﬁnal cross-modality encoder. Both these single-modality encoders uses a
self-attention and a feed-forward sub-layer, connected by residual blocks and layer-normalization.
In our work, we introduced changes to LXMERT to allow for video-question answering. To deal with
the challenge of comprehending temporal context along with cross-modal connections, we included
the SlowFast net here.  
For further details, refer to the project report at the end. 

## Requirements
Pytorch-1.3

## Quick Start
sh Run.sh

## References
1) 3D Attention is All You Need: https://www.academia.edu/41601920/3D_Attention_is_All_You_Need
2) LXMERT: https://arxiv.org/abs/1908.07490
3) SlowFast net: https://arxiv.org/abs/1812.03982
