This is the preprocessed version of the MPQA 1.2 dataset used
in the paper "Opinion Mining with Deep Recurrent Neural Networks"
by O. Irsoy and C. Cardie. You should cite the following if you
use the data:
"Annotating expressions of opinions and emotions in language"
Janyce Wiebe, Theresa Wilson, and Claire Cardie (2005). 
Language Resources and Evaluation, volume 39, issue 2-3, pp. 165-210.

-----------

dse.txt and ese.txt contain the actual sentences labeled with DSE and
ESE tags respectively. Sentences are separated by a newline. 
Every token is represented as three columns, separated with a tab.
Second column is the POS tags from Stanfor parser, which are unused
by RNNs but used by CRFs.
Third column is the output label (B, I or O) for the token.

sentenceid.txt shows which sentence belongs do which document. 

datasplit files show train/test partitions over documents. 

Dev set is defined as the documents which is listed in 
doclist.mpqaOriginalSubset file MINUS the training and test 
sentences over a fold (every fold uses the same dev set).

Therefore I have not used the entire MPQA2.0 data for the 
experiments (and used only the subset defined in 
doclist.mpqaOriginalSubset). This is because I wanted it to be 
comparable to the older results on MPQA1.2. 
I would suggest using the more recent version for new experiments.

