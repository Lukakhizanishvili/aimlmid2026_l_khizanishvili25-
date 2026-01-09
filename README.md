I and ML for Cybersecurity – Midterm Exam (Jan 9, 2026)
Student: Luka Khizanishvili
Repository: aimlmid2026_l_khizanishvili25

This repository contains my solutions for the Midterm Exam tasks described in the exam instructions provided by the course instructor.
All code is written in Python, and the work is reproducible by following the steps below.

Repository structure
midterm.py – single Python file containing solutions for both tasks

correlation/data_points.csv – manually collected (x, y) points from the online graph

spam_classifier/email_features.csv – provided dataset file (features + class label)

correlation/scatter.png – correlation visualization output

spam_classifier/ outputs:

model.joblib – trained logistic regression model
columns.json – feature column names used by the model
class_distribution.png – visualization 1
confusion_matrix.png – visualization 2
top_coefficients.png – visualization 3 (feature importance)
Task 1 – Finding the Correlation (10 points)
Data collection process
I opened the online graph at:

max.ge/aiml_midterm/97162_html
The data is displayed as blue dots. By hovering over each dot, the graph shows the exact coordinates of the data point.
I manually recorded the coordinates into the following file:

correlation/data_points.csv
Pearson correlation
To compute Pearson correlation coefficient (r) and generate a scatter plot, run:

python midterm.py correlation --points correlation/data_points.csv --out correlation/scatter.png
