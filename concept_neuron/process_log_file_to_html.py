"""
2018, University of Freiburg.
Joao Carvalho <carvalhj@cs.uni-freiburg.de>

Raw script (fast coded) to generate html tables from log files resulting
from training the logistic regression classifiers.
"""
import re


log_file = 'results/group_tags_nltk_data_1000/log.txt'
table_entry = '<tr>\n'
with open(log_file, 'r') as f:
    line = f.readline()
    while(True):
        if not line:
            break

        x = re.search(r'-----> CONCEPT: (?P<concept>.*)', line)

        if x:
            line = f.readline()
            if re.search(r'concept not found', line):
                continue

            table_entry += '\t<th style="text-align:center;">' + \
                x.group('concept') + '</th>\n'
            table_entry += '\t<td style="text-align:left;">' + \
                x.group('concept') + ' <br> ' + x.group('concept') + \
                ' - NOT</td>\n'

        y = re.search('Dataset statistics', line)
        if y:
            line = f.readline()
            line = f.readline()
            count_not = 0
            count_true = 0
            # trY
            x = re.search(r'.-+NOT+\s+-\s+(?P<count_not>\d+)', line)
            count_not += int(x.group('count_not'))
            line = f.readline()
            x = re.search(r'[A-Za-z,.()\$`:\']+\s+-\s+(?P<count_true>\d+)',
                          line)
            if x is not None:
                count_true += int(x.group('count_true'))
                line = f.readline()
            # teY
            line = f.readline()
            x = re.search(r'.-+NOT+\s+-\s+(?P<count_not>\d+)', line)
            count_not += int(x.group('count_not'))
            line = f.readline()
            x = re.search(r'[A-Za-z,.()\$`:\']+\s+-\s+(?P<count_true>\d+)',
                          line)
            if x is not None:
                count_true += int(x.group('count_true'))
                line = f.readline()
            # vaY
            line = f.readline()
            x = re.search(r'.-+NOT+\s+-\s+(?P<count_not>\d+)', line)
            count_not += int(x.group('count_not'))
            line = f.readline()
            x = re.search(r'[A-Za-z,.()\$`:\']+\s+-\s+(?P<count_true>\d+)',
                          line)
            if x is not None:
                count_true += int(x.group('count_true'))

            table_entry += '\t<td style="text-align:right;"> ' + \
                ' {} - {:.3f}'.format(count_true, count_true /
                                      (count_true + count_not)) + ' <br> ' + \
                ' {} - {:.3f}'.format(count_not, count_not /
                                      (count_true + count_not)) + '</td>'

        z = re.search('--- ALL dimensions', line)
        if z:
            for _ in range(10):
                line = f.readline()

            # test accuracy
            line = f.readline()
            w = re.search(r'Test accuracy:\s+(?P<acc>[+-]?([0-9]*[.])?[0-9]+)',
                          line)
            acc = w.group('acc')
            # test precision
            line = f.readline()
            w = re.search(
                r'Test precision:\s+(?P<prec>[+-]?([0-9]*[.])?[0-9]+)', line)
            prec = w.group('prec')
            # test recall
            line = f.readline()
            w = re.search(
                r'Test recall:\s+(?P<recall>[+-]?([0-9]*[.])?[0-9]+)', line)
            recall = w.group('recall')

            table_entry += \
                '\n\t<td style="text-align:center;">{} | {} | {}</td>\n' \
                .format(acc, prec, recall)

            line = f.readline()
            # features used
            line = f.readline()
            w = re.search(r'Features used:\s+(?P<features>\d+)', line)
            features = w.group('features')
            table_entry += '\t<td style="text-align:center;">{}</td>\n' \
                .format(features)

            # top features
            line = f.readline()
            w = re.search(r'Largest features:\s+(?P<largest_features>\[.*\])',
                          line)
            largest_features = w.group('largest_features')
            table_entry += '\t<td style="text-align:center;">{}</td>\n' \
                .format(largest_features)

        z = re.search('--- TOP (?P<k>\d) dimensions', line)
        if z:
            for _ in range(10):
                line = f.readline()

            # test accuracy
            line = f.readline()
            w = re.search(r'Test accuracy:\s+(?P<acc>[+-]?([0-9]*[.])?[0-9]+)',
                          line)
            acc = w.group('acc')
            # test precision
            line = f.readline()
            w = re.search(
                r'Test precision:\s+(?P<prec>[+-]?([0-9]*[.])?[0-9]+)', line)
            prec = w.group('prec')
            # test recall
            line = f.readline()
            w = re.search(
                r'Test recall:\s+(?P<recall>[+-]?([0-9]*[.])?[0-9]+)', line)
            recall = w.group('recall')

            table_entry += \
                '\t<td style="text-align:center;">{} | {} | {}</td>\n' \
                .format(acc, prec, recall)

            if z.group('k') == str(3):
                table_entry += '</tr>\n<tr>\n'

        line = f.readline()

print(table_entry)
