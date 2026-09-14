## Overall verdict

 

**Recommendation: REJECT in its current form.**

 

If this were submitted to **3DV/CVPR/T-PAMI**, I would expect a strong reject. For **Automation in Construction**, the application is relevant, but the current manuscript still has a serious novelty/evaluation problem and would likely be rejected unless substantially reworked.

 

The central issue is not that the pipeline is technically useless. It is that **almost every individual component is already established, while the one component presented as novel—torus-based elbow fitting—is demonstrably not new**. Worse, the experimental design does not currently validate the claimed industrial contribution because the principal real dataset, PSNet5, does not contain the elbow annotations needed to test the central claim.

 

The most damaging summary is:

 

> **You have a conventional semantic-segmentation → clustering → RANSAC → geometric fitting pipeline, with a standard torus primitive inserted at the end, and the only dataset on which the claimed elbow segmentation could matter does not actually test elbow segmentation.**

 

That is a very difficult novelty and validity position to defend.

 

---

 

# 1. Genuine novelty: is DL + Torus-RANSAC actually novel?

 

### Short answer: **No, not as currently formulated.**

 

The architecture

  

is a well-established scan-to-BIM / industrial point-cloud paradigm.

 

More importantly, **torus fitting itself is old**.

 

Schnabel et al.'s Efficient RANSAC explicitly detects **planes, spheres, cylinders, cones and tori** from unorganized point clouds. This is not a theoretical possibility—it is the original purpose of the algorithm. [Wiley Online Library+1](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2007.01016.x?utm_source=chatgpt.com)

 

Modern CGAL still exposes torus detection as part of its Efficient RANSAC implementation, with explicit torus axis, center, major radius and minor radius parameters. [CGAL Manual+1](https://doc.cgal.org/latest/Shape_detection/classCGAL_1_1Shape__detection_1_1Torus.html?utm_source=chatgpt.com)

 

There are also considerably closer works than generic primitive detection.

 

### Particularly damaging precedent #1: elbow-specific torus geometry

 

Chan et al., *Geometric Modelling for 3D Point Clouds of Elbow Joints in Piping Systems*, Sensors 2020, explicitly models **45° and 90° pipe elbows as torus-like structures** and estimates their geometric parameters from point clouds using nonlinear least-squares/Gauss–Helmert adjustment. They report simulated and real experiments, including diameter accuracy of 97.2%. [MDPI+1](https://www.mdpi.com/1424-8220/20/16/4594?utm_source=chatgpt.com)

 

This is extremely close to your claimed contribution.

 

Indeed, the paper explicitly says:

 

 - elbow joints are torus-like;
 - 90° and 45° elbows are modeled;
 - translation, rotation and dimensional parameters are estimated;
 - incomplete scans can be handled;
 - real laser-scanned data are evaluated. [PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7471979/?utm_source=chatgpt.com)

 

Therefore:

 

> **"We introduce torus fitting for elbow radius estimation" cannot be claimed as a novel contribution.**

 

At most, you can claim a **different robust estimation implementation**.

 

But then you need to prove that your estimator is materially better.

 

---

 

### Particularly damaging precedent #2: torus fitting after segmentation

 

Raffo et al., *Fitting and recognition of geometric primitives in segmented 3D point clouds using a localized voting procedure*, Computer-Aided Geometric Design 2022, explicitly handles **cylinders and tori**, including initialization, canonicalization, fitting and recognition from segmented point clouds. It is explicitly designed for noise, missing parts and outliers and evaluates synthetic and industrial scans. [ScienceDirect+1](https://www.sciencedirect.com/science/article/pii/S0167839622000590?utm_source=chatgpt.com)

 

The paper's workflow is conceptually:

  

That is dangerously close to your:

  

---

 

### Particularly damaging precedent #3: generic primitive fitting benchmarks already contain torus

 

Fit4CAD explicitly benchmarks fitting of **planes, cylinders, cones, spheres and tori**, with ground-truth implicit and parametric representations and primitive-fitting accuracy metrics. [ScienceDirect+1](https://www.sciencedirect.com/science/article/abs/pii/S0097849321002053?utm_source=chatgpt.com)

 

So a claim such as:

 

> "We propose a robust torus-fitting algorithm for point clouds."

 

would be difficult to sustain without demonstrating a genuinely new estimator, not merely:

 

 - RANSAC,
 - PCA,
 - torus equation,
 - nonlinear refinement,
 - parameter bounds,
 - outlier filtering.

 

Those are engineering modifications, not au[ScienceDirect+1](https://www.sciencedirect.com/science/article/pii/S1474034620300902?utm_source=chatgpt.com)[ScienceDirect+1](https://www.sciencedirect.com/science/article/pii/S0926580521003253?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580522006057?utm_source=chatgpt.com)[GitHub](https://github.com/Xieyuan0018/PipeNet-data?utm_source=chatgpt.com)[Google Patents+1](https://patents.google.com/patent/WO2023096579A2/en?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1474034620300902?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580521003253?utm_source=chatgpt.com)[GitHub+1](https://github.com/PointCloudYC/ResPointNet2?utm_source=chatgpt.com)[GitHub+1](https://github.com/pointcloudyc/Industrial3D?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2603.28660?utm_source=chatgpt.com)[GitHub](https://github.com/pointcloudyc/Industrial3D?utm_source=chatgpt.com)[GitHub](https://github.com/pointcloudyc/Industrial3D?utm_source=chatgpt.com)[GitHub+1](https://github.com/pointcloudyc/Industrial3D?utm_source=chatgpt.com)[PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7471979/?utm_source=chatgpt.com)[DOI](https://doi.org/10.3390/app15063397?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580523003801?utm_source=chatgpt.com)[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.html?utm_source=chatgpt.com)[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Hierarchical_Aggregation_for_3D_Instance_Segmentation_ICCV_2021_paper.html?utm_source=chatgpt.com)[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.html?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2210.03105?utm_source=chatgpt.com)[GitHub](https://github.com/pointcloudyc/Industrial3D?utm_source=chatgpt.com)[MDPI](https://www.mdpi.com/1424-8220/20/16/4594?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580521003253?utm_source=chatgpt.com)[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.html?utm_source=chatgpt.com)[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Vu_SoftGroup_for_3D_Instance_Segmentation_on_Point_Clouds_CVPR_2022_paper.html?utm_source=chatgpt.com)[Wiley Online Library](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2007.01016.x?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167839622000590?utm_source=chatgpt.com)[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Supervised_Fitting_of_Geometric_Primitives_to_3D_Point_Clouds_CVPR_2019_paper.html?utm_source=chatgpt.com)[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_PrimitiveNet_Primitive_Instance_Segmentation_With_Local_Primitive_Embedding_Under_Adversarial_ICCV_2021_paper.html?utm_source=chatgpt.com)[MDPI](https://www.mdpi.com/1424-8220/20/16/4594?utm_source=chatgpt.com)[DOI](https://doi.org/10.3390/app15063397?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580523000171?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2603.28660?utm_source=chatgpt.com)[GitHub](https://github.com/pointcloudyc/Industrial3D?utm_source=chatgpt.com)[OpenReview](https://openreview.net/pdf?id=3Pbra-_u76D&utm_source=chatgpt.com)[Wiley Online Library+2MDPI+2](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2007.01016.x?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926580521003253?utm_source=chatgpt.com)[arXiv+3CVF Open Access+3CVF Open Access+3](https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_PointGroup_Dual-Set_Point_Grouping_for_3D_Instance_Segmentation_CVPR_2020_paper.html?utm_source=chatgpt.com)
