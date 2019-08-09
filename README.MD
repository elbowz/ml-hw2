# Machine Learning: MLP using sklearn  
### more detailed information in ITALIAN: [REPORT.MD](REPORT.MD)  

Comparing different Multi Layer Perceptron **architecture**.

## Utilized libs 
    
* **pandas**:  load and manipulation of **DataSet** 
* **sklearn**:  algo, stats and splitting of **TrainingSet**
* **matplotlib**: graph plotting
  
## Script usage  
  
```bash  
main.py [wine|breast-cancer|letters|poker][test_size < 1 | k-fold > 1][-v]  
```  
  
* **dataset**: [wine|breast-cancer|letters|poker] - select DataSet. Default is *wine*  
* **test_size**: [0,1] - TestSet dimension  
* **k-fold**: {0, 1, 2, 3, 4, 5...} - K fold iteration  
* **-v**: enable verbose mode  

## DataSets

The **DataSet** taken from [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/index.php) are: 
* [Wine](https://archive.ics.uci.edu/ml/datasets/Wine)
* [Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
* [Letter Recognition](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
* [Poker Hand](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)