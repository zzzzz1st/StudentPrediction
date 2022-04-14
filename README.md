<img src="https://user-images.githubusercontent.com/52376408/163346280-a274804c-e071-4ce1-9462-59f577d38b9a.png" width="30%" height="30%">

# StudentPrediction
Predicting Academic Performance Of Students

1. Please go to the https://analyse.kmi.open.ac.uk/open_dataset website, download the dataset and put it inside the project file. Check for the similarity of the filenames between the script and csv files.


2. Change the sample configuration (inside preprocess.py) in order to select the total number of students. It's set on 1000 by default but it takes time to produce models.
```python
# TOTAL NUMBER OF STUDENTS
st_info = st_info.sample(1000)
```
3. Run the preprocess script to produce .pkl files.
```bash
python preprocess.py
```
4. Run the analysis script to see the scores
```bash
python analysis.py
```


