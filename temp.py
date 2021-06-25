import csv

with open('book30-listing-train.csv', mode='r',encoding="utf8", errors='ignore') as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=',')
    f = open('BookCover30_text/train_data.txt','w')
    line_count = 1
    for lines in csv_reader :
        f.write(lines[3] + '\n')
        line_count+=1
    f.close()

print('process completed\n')
