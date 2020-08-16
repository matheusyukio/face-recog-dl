
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import pandas as pd

def write_cm_report(Y, classes, predict, filename):
	print('Confusion Matrix')
	cm = confusion_matrix(classes, predict)
	file = open(filename, 'a+')
	file.write('cm \n')
	file.write(str(cm))
	file.write('\n')
	average_type = ['micro','macro']
	for i in average_type:
	    score = f1_score(classes, predict, average=i)
	    precision = precision_score(classes, predict, average=i)
	    recall = recall_score(classes, predict, average=i)
	    file.write('f1 score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(score, precision, recall))
	    file.write('\n')
	file.write("Accuracy: %f" % accuracy_score(classes, predict))
	file.write('\n')

	nomes_classes = []
	for i in pd.DataFrame(Y.groupby('name')['name'].nunique().reset_index(name="unique"))[
	    'name']:
	    nomes_classes.append(str(i))
	file.write(str(classification_report(classes, predict, target_names=nomes_classes)))
	file.write('\n')
	file.close()
