
This is a sample implementation of the method in CVPR paper:

Reference: Jia-Bin Huang and Ming-Hsuan Yang, "Fast Sparse Representation with Prototypes.", the 23th IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR 10'), San Francisco, CA, USA, June 2010.

Contact: For any questions, email me by jbhuang0604@gmail.com


Directories===============================================


/tools/
This directory contains the tools that you can use in solving sparse coding and dicitonary learning.

/tools/sparse coding/
This directory contains several sparse coding algorithms, including

1. l1magic
http://www.acm.caltech.edu/l1magic/

2. SparseLab
http://sparselab.stanford.edu/

3. Efficient sparse coding
http://www.stanford.edu/~hllee/softwares/nips06-sparsecoding.htm

4. YALL1: Your ALgorithms for L1
http://www.caam.rice.edu/~optimization/L1/YALL1/

5. Smoothed L0
http://ee.sharif.edu/~SLzero/


/tools/dictionary learning/
This directory contains the K-SVD toolbox
http://www.cs.technion.ac.il/~elad/software/


Usage===============================================

1. Open run_sparsity_based_classification.m
2. Specify one sparse coding method (default: l1magic)
3. Specify one dataset your want to test (default: the extended Yale database B)
4. Run the script


Parameters===============================================


Contact

Any comments or questions, please contact

Feedback and comments are welcome.
Jia-Bin Huang (jbhuang0604@gmail.com)
Ming-Hsuan Yang (mhyang@ucmerced.edu)