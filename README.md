# advlearning_algos

> Comparing classic Adversarial Learning algorithms.

> A collaborative repo for class project issued by Dr. Ray in CSDS 440: Machine Learning during Fall 2020 semester.

---
## Plan

Hey guys, we been ghosting this project for a while — which I understood, weekly written/programming plus works from other classes can be tiring — but now we are kinda on the edge of being too late. Since most of you have not replied anything or reached out to me personally, I kinda made most of the "group decision" here; please let me know if there will be any major conflict.

### Timeline

Since the implementation requirement per person is 1 algorithm + 1 extension and we obviously have to do a white-up, I would say a reasonable timeline would be:

* Finish your code implementation of your base algorithm (with performance evaluation) before Saturday (11/28/2020).
* Finish your extension algorithm (with performance evaluation) before Sunday (11/29/2020).
* Finish your write-up for Group Report before next Tuesday (12/1).

Please let me know if you can't make it for whatever reason.

### Dataset

It seems most adversarial learning work were deployed on the following 3 datasets, so I think we should stick with them.

* [MINIST](http://yann.lecun.com/exdb/mnist/)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [ImageNet](http://www.image-net.org)

### Library and Yak Shaving

Dr. Ray said we can use whatever libraries we'd like, but there will be penalty if we just `import` models wrote by others. But yak shaving can be very hard, especially for adversarial learning there will be a lot of engineering work regarding getting the original benchmark, feeding the attack model, getting the adversarial benchmark, feeding the defense model, getting the defensive benchmark... etc. The required work can be double/tripled with multiple dataset, classifiers...

So me and my partner actually discovered this awesome library called [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox). It has pretty much did all the engineering — getting dataset, getting classic classifiers/estimators, evaluate performance, etc. — in fact, this toolbox actually include many famous adversarial algorithm as well that you can import. Although we can't use the adversarial algorithm, it can be in our interests to learn from their implementations.

Me and my partner have piped this library and played around it for a bit, it works with some minor tweaking of TensorFlow (at least for Keras). So my plan is to use the fundamentals of this library, but implement the algorithms ourself. You may find anything that suits yourself, but this is probably our best bet as it includes many examples ([here](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/examples) and [here](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/notebooks)) that literally showing us how to pipeline the work — so all we have to do is to understand the algorithm, implement it with fundamentals of this library, channel it back into the pipeline, do some combination of attack/defense evaluation and we are basically done.



### Algorithms

So in adversarial learning we basically have the attack side and defense side, it would be nice if we can balance out our implementations a bit, but attacks are generally easier to do. What we have planned for now are:

| Name                                | Algorithm                  | Paper                                                                                                  | Attack/Defense | Method        | Dataset           | Example                                                                                                                                                                     | Note                                             |
|-------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------|----------------|---------------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| Henry                               | Fast Gradient Method       | [Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)                                             | Attack         | Evasion       | MINIST & CIFAR-10 | [adversarial_training_mnist.ipynb](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_training_mnist.ipynb)                       | * White-box                                      |
| Henry                               | Projected Gradient Descent | [Madry et al., 2017](https://arxiv.org/abs/1706.06083)                                                 | Attack         | Evasion       | MINIST & CIFAR-10 | [attack_defence_imagenet.ipynb](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/detection_adversarial_samples_cifar10.ipynb)               | * White-box                                      |
| Henry  (You can take it from me)    | Back Door                  | [Gu, et. al., 2017](https://arxiv.org/abs/1708.06733)                                                  | Attack         | Poisoning     | MINIST & CIFAR-10 | [poisoning_defense_neural_cleanse.ipynb](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_neural_cleanse.ipynb)           |                                                  |
| Mingyang  (You can take it from me) | Adversarial Trainer        | [Szegedy et al., 2013](http://arxiv.org/abs/1312.6199)                                                 | Defense        | Trainer       | MINIST & CIFAR-10 | [adversarial_training_mnist.ipynb](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/adversarial_training_mnist.ipynb)                       | * Or binary input detector * Proven against FGM. |
| Mingyang  (You can take it from me) | Binary Input Detector      | [Al-Dujaili et al., 2018](https://arxiv.org/abs/1801.02950)                                            | Defense        | Detector      | MINIST & CIFAR-10 | [detection_adversarial_samples_cifar10.ipynb](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/detection_adversarial_samples_cifar10.ipynb) | * Or adversarial trainer * Proven against FGM.   |
| Mingyang                            | Spatial Smoothing          | [Xu et al., 2017](http://arxiv.org/abs/1704.01155)                                                     | Defense        | Pre-Processor | MINIST & CIFAR-10 | [attack_defence_imagenet.ipynb](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/detection_adversarial_samples_cifar10.ipynb)               | * Proven against PGD.                            |
| Mingyang                            | Neural Cleanse             | [Wang et al., 2019](http://people.cs.uchicago.edu/~ravenben/publications/abstracts/backdoor-sp19.html) | Defense        | Transformer   | MINIST & CIFAR-10 | [poisoning_defense_neural_cleanse.ipynb](https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/poisoning_defense_neural_cleanse.ipynb)           | * Proven against Backdoor.                       |
| Alex                                |                            |                                                                                                        |                |               |                   |                                                                                                                                                                             |                                                  |
| Austin                              | Defensive Distillation     | [Papernot et al., 2015](https://arxiv.org/abs/1511.04508)                                              | Defense       | Transformer    | MINIST & CIFAR-10 |                                                                                                                                                                             |                                                  |
| David                               |                            |                                                                                                        |                |               |                   |                                                                                                                                                                             |                                                  |

**PLEASE FILL THIS OUT NO LATER THAN FRIDAY (11/27/2020) AS OTHERWISE I WILL ASSUME YOU WILL NOT BE CONTRIBUTING ANY WORK.** Me and my partner plan to throw in 3 attack algos and 3 defense algos, hopefully with some extensions, so we will have a minimum content for Group Report — but that will be done at the cost of quality, and our algos can't be part of your individual report.

Please let us know if you plan to contribute (and what will you working on). You may either find one algorithm I proposed and take it (the one saying `you can take it from me`), or you can research your own one.



Last, HAPPY HOLIDAY!

---
---
Since we have decided to migrate the collaboration to `GitHub` for better readability on texts/PDFs/images and potential useful features like *issues*. This repository will be mirrored with `csevcs` before the final deadline (12/04/2020).

To do the mirroring, simply follow:

```
your_work_dir $ git clone --bare https://github.com/choH/advlearning_algos.git
your_work_dir $ cd advlearning_algos.git
advlearning_algos $ git push --mirror https://csevcs.case.edu/git/2020_fall_440_advlearning
```

Note this maneuver will completely overwrite the repository on `csevcs` — which shouldn't be a big deal since all our developments are initiated here, but please be caution.

---
## Contribution

If you are not a member of *advlearning*, don't.

---
## Acknowledgement

* Austin Keppers `agk51`
* David Meshnick `dcm101`
* Minyang Tie `mxt479`
* Alex Useloff `adu3`
* Henry Zhong `sxz517` ([@choH](https://github.com/choH))
