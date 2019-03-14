Homework Assignment #7: Learning-based recognition
Zhenyu Yang
=========================================================================
Program running instruction:
1. Please ensure the folder "cifar-10-batches-py/" containing data sets is 
   in the same path as pythonfiles.

2. To start training, run:

        python3 prog7.py

   or
        python3 prog7_extra.py

3. To verify the resulted model on the test set, run:

	python3 prog72.py 

=========================================================================

Program description:

This code can run in python3.

prog7.py uses a network design in the homework description(http://www.cs.ucsb.edu/~cs181b/hw/prog7.pdf). This model achieved accuracy of 76.9% on the training set and accuracy of 60% on the test set.

prog7_extra.py uses a network design with an extra layer between layer 2 and layer 3 in in prog7.py. With the extra layer, this model achieved accuracy of 83% on the training set and accuracy of 65.7% on the test set.

prog72.py is used to verify the resulted model.
=========================================================================

